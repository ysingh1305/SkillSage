import os
import re
import io
import json
import sqlite3
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import fuzz, process
from dotenv import load_dotenv

# Load environment variables from backend/.env
load_dotenv()

# --- LangChain / OpenAI ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy

# --- Resume readers ---
import pdfplumber
import docx2txt
from werkzeug.utils import secure_filename

# =========================
# Config & Environment
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Put it in backend/.env (no quotes) and restart."
    )

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

TOP_K = int(os.getenv("TOP_K", 8))
SEM_THRESHOLD = float(os.getenv("SEM_THRESHOLD", 0.55))
ALIAS_FUZZ_THRESHOLD = int(os.getenv("ALIAS_FUZZ_THRESHOLD", 88))

USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "true").lower() == "true"
USE_LLM_ROLE = os.getenv("USE_LLM_ROLE", "false").lower() == "true"

HERE = os.path.dirname(__file__)
SKILLS_PATH = os.path.join(HERE, "skills.json")
STACKS_PATH = os.path.join(HERE, "stacks.json")  # optional clusters/relationships
DB_PATH = os.path.join(HERE, "skillsage.db")

ALLOWED_EXTS = {".pdf", ".docx", ".txt"}


# =========================
# Load Skills (+ optional stacks)
# =========================
with open(SKILLS_PATH, "r", encoding="utf-8") as f:
    SKILLS: List[Dict[str, Any]] = json.load(f)

STACKS = []
if os.path.exists(STACKS_PATH):
    with open(STACKS_PATH, "r", encoding="utf-8") as f:
        STACKS = json.load(f)

skill_texts = [f'{s["name"]} — {s.get("blurb", "")}' for s in SKILLS]
metas = [{"idx": i, **SKILLS[i]} for i in range(len(SKILLS))]

# =========================
# Build Embeddings & Vector Store (FAISS)
# =========================
emb = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
vs = FAISS.from_texts(
    texts=skill_texts,
    embedding=emb,
    metadatas=metas,
    distance_strategy=DistanceStrategy.COSINE,
)
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)

# =========================
# SQLite for tracking stats
# =========================
def db_init():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_hint TEXT,
            industry_hint TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS job_skills (
            job_id INTEGER,
            skill TEXT,
            level TEXT,
            years REAL,
            required INTEGER,
            confidence REAL,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )"""
    )
    con.commit()
    con.close()

def db_new_job(role_hint: str, industry_hint: str) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO jobs(role_hint, industry_hint) VALUES (?, ?)",
        (role_hint, industry_hint),
    )
    con.commit()
    jid = cur.lastrowid
    con.close()
    return jid

def db_log(job_id: int, rows: List[Tuple[str, str, float, int, float]]):
    rows_with_id = [(job_id, *r) for r in rows]
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executemany(
        "INSERT INTO job_skills(job_id,skill,level,years,required,confidence) VALUES (?,?,?,?,?,?)",
        rows_with_id,
    )
    con.commit()
    con.close()

db_init()

# =========================
# Heuristics: levels / years / required vs preferred
# =========================
LEVEL_MAP = {
    "expert": ["expert", "advanced", "senior", "deep", "strong"],
    "intermediate": ["intermediate", "proficient", "solid", "working knowledge"],
    "beginner": ["basic", "familiarity", "exposure", "foundational", "entry"],
}
LEVEL_ORDER = ["beginner", "intermediate", "expert"]

RE_REQ = re.compile(r"\b(must\s*have|required|need(ed)?|minimum|at\s*least)\b", re.I)
RE_PREF = re.compile(r"\b(preferred|nice\s*to\s*have|bonus|plus)\b", re.I)
RE_YEARS = re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs)\b", re.I)

def detect_level_and_years(jd: str, skill: str) -> Tuple[str, float]:
    text = jd.lower()
    years = 0.0
    for m in RE_YEARS.finditer(text):
        try:
            years = max(years, float(m.group(1)))
        except Exception:
            pass
    best = None
    for lvl, keys in LEVEL_MAP.items():
        for k in keys:
            if k in text:
                if best is None or LEVEL_ORDER.index(lvl) > LEVEL_ORDER.index(best):
                    best = lvl
    return (best or "unspecified", years)

def detect_required_preferred(jd: str, skill: str) -> str:
    lower = jd.lower()
    idx = lower.find(skill.lower())
    window_text = lower[max(idx - 120, 0) : idx + 120] if idx != -1 else lower
    if RE_REQ.search(window_text): return "required"
    if RE_PREF.search(window_text): return "preferred"
    if RE_REQ.search(lower): return "required"
    if RE_PREF.search(lower): return "preferred"
    return "unspecified"

def llm_required_preferred(jd: str, skill: str) -> str:
    prompt = f"""Classify if the skill "{skill}" in the job description is REQUIRED or PREFERRED.
Return exactly one word: "required" or "preferred". If unclear, return "unspecified".

Job description:
{jd}"""
    try:
        out = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        if "required" in out: return "required"
        if "preferred" in out: return "preferred"
        return "unspecified"
    except Exception:
        return "unspecified"

# =========================
# Alias matching
# =========================
def alias_hits(text: str):
    hits = set()
    lower = text.lower()
    for i, s in enumerate(SKILLS):
        names = [s["name"], *s.get("aliases", [])]
        for alias in names:
            a = alias.lower()
            if len(a) >= 3 and a in lower:
                hits.add(i); break
        if i not in hits:
            best, score, _ = process.extractOne(lower, names, scorer=fuzz.partial_ratio)
            if score >= ALIAS_FUZZ_THRESHOLD: hits.add(i)
    return hits

# =========================
# Snippet utilities
# =========================
EEO_RE = re.compile(r"(equal employment|eeo|discriminat|veteran|disabilit|race|religion)", re.I)

def _split_blocks(text: str) -> List[str]:
    lines = text.splitlines()
    blocks, buf = [], []
    bullet_pat = re.compile(r"^\s*[-*•]\s+")
    for ln in lines:
        if bullet_pat.match(ln):
            if buf: blocks.append(" ".join(buf).strip()); buf = []
            blocks.append(ln.strip())
        elif ln.strip() == "":
            if buf: blocks.append(" ".join(buf).strip()); buf = []
        else:
            buf.append(ln.strip())
    if buf: blocks.append(" ".join(buf).strip())
    return [b for b in blocks if len(b) > 20 and not EEO_RE.search(b)]

def _split_sentences(block: str) -> List[str]:
    blk = re.sub(r"\s+", " ", block.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", blk)
    return [p.strip(" -") for p in parts if len(p.strip()) > 3]

def _contains_alias(text: str, names: List[str]) -> bool:
    low = text.lower()
    for a in names:
        a = a.lower().strip()
        if len(a) >= 3 and a in low:
            return True
    return False

def extract_snippets(text: str, needle: str, window_sentences: int = 1) -> List[str]:
    snippets: List[str] = []
    names = [needle]
    try:
        skill = next((s for s in SKILLS if s["name"].lower() == needle.lower()), None)
        if skill: names = [skill["name"], *skill.get("aliases", [])]
    except Exception:
        pass
    blocks = _split_blocks(text)
    for b in blocks:
        if _contains_alias(b, names) and len(snippets) < 3:
            snippets.append(b)
    if len(snippets) < 3:
        for b in blocks:
            sents = _split_sentences(b)
            for i, s in enumerate(sents):
                if _contains_alias(s, names):
                    start = max(0, i - window_sentences)
                    end = min(len(sents), i + window_sentences + 1)
                    chunk = " ".join(sents[start:end]).strip()
                    if chunk and not EEO_RE.search(chunk):
                        snippets.append(chunk)
                        if len(snippets) >= 3: break
            if len(snippets) >= 3: break
    clean, seen = [], set()
    for sn in snippets:
        sn = re.sub(r"\s+", " ", sn).strip(" .,-")
        if sn and sn.lower() not in seen:
            clean.append(sn); seen.add(sn.lower())
    return clean[:3]

# =========================
# Role/Industry detection (title-aware + LLM fallback)
# =========================
ROLE_CATALOG = [
    "data analyst", "data scientist", "machine learning engineer", "data engineer",
    "analytics engineer", "backend engineer", "frontend engineer", "full stack engineer",
    "devops engineer", "cloud engineer", "security engineer",
    "business analyst", "product manager"
]
INDUSTRY_CATALOG = [
    "consulting", "finance", "healthcare", "ecommerce", "saas", "gaming",
    "education", "government", "telecom", "manufacturing", "retail", "media"
]

ROLE_LEXICON = {
    "data analyst": {
        "title": ["data analyst", "analytics", "bi analyst"],
        "hard":  ["sql","tableau","power bi","excel","dashboard","visualization","report","ad hoc","kpi","insight"],
        "soft":  ["stakeholder","requirements","business questions","storytelling"]
    },
    "data scientist": {
        "title": ["data scientist","applied scientist"],
        "hard":  ["modeling","machine learning","ml","scikit-learn","xgboost","pytorch","tensorflow","notebook","feature engineering"],
        "soft":  ["experimentation","ab testing","hypothesis"]
    },
    "machine learning engineer": {
        "title": ["ml engineer","machine learning engineer"],
        "hard":  ["mlops","inference","serving","feature store","pytorch","tensorflow","latency","throughput","monitoring","drift"],
        "soft":  ["production","systems"]
    },
    "data engineer": {
        "title": ["data engineer","etl developer"],
        "hard":  ["airflow","spark","kafka","dbt","warehouse","pipeline","ingestion","elt","lakehouse","bigquery","snowflake"],
        "soft":  ["reliable","scalable"]
    },
    "analytics engineer": {
        "title": ["analytics engineer"],
        "hard":  ["dbt","sql","dimensional modeling","semantic layer","metrics","warehouse","data modeling"],
        "soft":  ["stakeholder","documentation"]
    },
    "backend engineer": {
        "title": ["backend engineer","server engineer","api engineer"],
        "hard":  ["api","microservice","rest","grpc","flask","fastapi","django","spring","java","golang","node","express","database"],
        "soft":  ["scalability","reliability"]
    },
    "frontend engineer": {
        "title": ["frontend engineer","ui engineer","web engineer"],
        "hard":  ["react","next.js","typescript","javascript","vue","css","html","accessibility","component"],
        "soft":  ["design system","ux"]
    },
    "full stack engineer": {
        "title": ["full stack","full-stack"],
        "hard":  ["react","next.js","node","express","django","flask","sql","api"],
        "soft":  []
    },
    "devops engineer": {
        "title": ["devops","sre","site reliability"],
        "hard":  ["kubernetes","k8s","terraform","helm","argo","cicd","docker","observability","prometheus","grafana"],
        "soft":  ["incident","on-call"]
    },
    "cloud engineer": {
        "title": ["cloud engineer"],
        "hard":  ["aws","gcp","azure","iam","vpc","lambda","ecs","eks","gke","cloud run"],
        "soft":  []
    },
    "security engineer": {
        "title": ["security engineer","appsec"],
        "hard":  ["oauth","oidc","iam","vulnerability","owasp","sast","dast","threat"],
        "soft":  []
    },
    "business analyst": {
        "title": ["business analyst"],
        "hard":  ["requirements","process","kpi","excel","sql","report"],
        "soft":  ["stakeholder","workshops"]
    },
    "product manager": {
        "title": ["product manager","pm"],
        "hard":  ["roadmap","prioritize","metrics","experiments","requirements"],
        "soft":  ["stakeholder","strategy"]
    }
}

INDUSTRY_LEXICON = {
    "consulting":   ["client","stakeholder","engagement","deliverable","consulting","pwc","deloitte","kpmg","ey"],
    "finance":      ["portfolio","trading","risk","bank","loan","fintech","credit"],
    "healthcare":   ["patient","clinical","emr","fda","hipaa","pharma","ehr"],
    "ecommerce":    ["cart","checkout","merchant","conversion","catalog","marketplace"],
    "saas":         ["subscription","tenant","b2b","multi-tenant","sso"],
    "gaming":       ["game","unity","unreal","liveops"],
    "education":    ["student","curriculum","edtech","lms"],
    "government":   ["public sector","federal","state","regulation","rfx"],
    "telecom":      ["network","5g","billing","subscriber"],
    "manufacturing":["factory","supply chain","mes","plm"],
    "retail":       ["store","pos","merchandising","inventory"],
    "media":        ["streaming","adtech","impressions","content"]
}

def _score_role_heuristic(jd: str) -> Tuple[str, float]:
    text = jd.lower()
    first = jd.splitlines()[0].lower() if jd.strip() else ""
    def count_all(keys, hay):
        return sum(hay.count(k) for k in keys)
    best_role, best_score = "", 0.0
    for role, bag in ROLE_LEXICON.items():
        score = 0.0
        score += 5.0 * count_all(bag["title"], first)
        score += 1.5 * count_all(bag["hard"], text)
        score += 0.5 * count_all(bag["soft"], text)
        if score > best_score:
            best_role, best_score = role, score
    return best_role, best_score

def _score_industry_heuristic(jd: str) -> Tuple[str, float]:
    text = jd.lower()
    best_ind, best = "", 0
    for ind, keys in INDUSTRY_LEXICON.items():
        c = sum(text.count(k) for k in keys)
        if c > best:
            best_ind, best = ind, c
    return best_ind, float(best)

def _classify_role_industry_llm(jd: str) -> Tuple[str, str, float]:
    prompt = f"""
You are a classifier. Choose ONE role and ONE industry from the lists below that best fit the job description.
If uncertain, return "unspecified" for that field.

ROLES: {", ".join(ROLE_CATALOG)}
INDUSTRIES: {", ".join(INDUSTRY_CATALOG)}

Return STRICT JSON with keys role, industry, confidence (0..1). No prose.

JD:
{jd}
"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        import json as _json, re as _re
        m = _re.search(r"\{.*\}", resp, _re.S)
        data = _json.loads(m.group(0)) if m else _json.loads(resp)
        role = str(data.get("role","unspecified")).lower()
        industry = str(data.get("industry","unspecified")).lower()
        conf = float(data.get("confidence", 0.0))
        role = role if role in ROLE_CATALOG else "unspecified"
        industry = industry if industry in INDUSTRY_CATALOG else "unspecified"
        return role, industry, conf
    except Exception:
        return "unspecified", "unspecified", 0.0

def detect_role_industry(jd: str) -> Tuple[str, str]:
    r_heur, r_score = _score_role_heuristic(jd)
    i_heur, i_score = _score_industry_heuristic(jd)

    heur_conf_role = 1.0 if r_score >= 8 else 0.0
    heur_conf_ind  = 1.0 if i_score >= 2 else 0.0

    role = r_heur if heur_conf_role else ""
    industry = i_heur if heur_conf_ind else ""

    if USE_LLM_ROLE and (not role or not industry):
        r_llm, i_llm, conf = _classify_role_industry_llm(jd)
        if not role and conf >= 0.5:
            role = r_llm
        if not industry and conf >= 0.5:
            industry = i_llm

    return role or "unspecified", industry or "unspecified"

# =========================
# Relationships / suggestions
# =========================
SUGGEST_MAP = {s["name"]: s.get("suggest", []) for s in SKILLS}
GROUP_INDEX = defaultdict(list)
for st in STACKS:
    for sk in st.get("skills", []):
        GROUP_INDEX[sk].append(st["name"])

def rollup_groups(detected_names: List[str]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for name in detected_names:
        for g in GROUP_INDEX.get(name, []):
            groups[g].append(name)
    return [{"group": g, "skills": sorted(v)} for g, v in groups.items()]

def missing_suggestions(detected_names: List[str]) -> List[str]:
    need = set(); present = set(detected_names)
    for n in detected_names:
        for s in SUGGEST_MAP.get(n, []):
            if s not in present: need.add(s)
    return sorted(need)

# =========================
# Utility
# =========================
def _to_01(raw: float) -> float:
    """
    Map raw FAISS score to [0,1].
    If it looks like cosine similarity [-1,1] → (s+1)/2.
    Else treat as cosine distance [0,2]       → 1 - d/2.
    """
    try:
        x = float(raw)
    except Exception:
        return 0.0
    if -1.0 <= x <= 1.0:
        return max(0.0, min(1.0, (x + 1.0) / 2.0))
    if x >= 0.0:
        d = max(0.0, min(2.0, x))
        return 1.0 - d / 2.0
    return 0.0

def read_resume_file(file_storage) -> str:
    filename = secure_filename(file_storage.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    data = file_storage.read()
    if not ext or ext not in {".pdf", ".docx", ".txt"}:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")
    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")
    if ext == ".docx":
        tmp = os.path.join(HERE, "_tmp_upload.docx")
        with open(tmp, "wb") as f: f.write(data)
        try: return docx2txt.process(tmp) or ""
        finally:
            try: os.remove(tmp)
            except: pass
    if ext == ".pdf":
        text = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    return ""

def extract_skills_from_text(text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    docs = vs.similarity_search_with_score(text, k=top_k)
    results: Dict[int, Dict[str, Any]] = {}
    for doc, raw in docs:
        score = _to_01(float(raw))
        if score >= SEM_THRESHOLD:
            meta = doc.metadata; idx = int(meta["idx"])
            results[idx] = {
                "name": meta.get("name"),
                "category": meta.get("category"),
                "blurb": meta.get("blurb",""),
                "examples": meta.get("examples", []),
                "score_semantic": round(score, 4),
                "snippets": [],
            }
    alias_idx = alias_hits(text)
    for idx in alias_idx:
        s = SKILLS[idx]
        entry = results.get(idx, {
            "name": s["name"], "category": s.get("category"),
            "blurb": s.get("blurb",""), "examples": s.get("examples", []),
            "score_semantic": None, "snippets": []
        })
        for needle in [s["name"], *s.get("aliases", [])]:
            entry["snippets"].extend(extract_snippets(text, needle))
        entry["snippets"] = list(dict.fromkeys(entry["snippets"]))
        results[idx] = entry
    ranked = sorted(results.values(), key=lambda r:(r["score_semantic"] or 0.0), reverse=True)
    out = []
    for r in ranked:
        lvl, yrs = detect_level_and_years(text, r["name"])
        r["level"] = lvl; r["years"] = yrs if yrs > 0 else None
        # NOTE: No snippet-based filtering here (per your request not to affect resume flow)
        out.append(r)
    return out

def _level_score(resume_level: str, jd_level: str) -> float:
    if jd_level == "unspecified" or not jd_level: return 0.5
    if resume_level == "unspecified" or not resume_level: return 0.25
    try:
        r = LEVEL_ORDER.index(resume_level); j = LEVEL_ORDER.index(jd_level)
    except ValueError:
        return 0.25
    if r > j: return 1.0
    if r == j: return 0.7
    return 0.0

def _years_score(resume_years: float|None, jd_years: float|None) -> float:
    if not jd_years: return 0.5
    if not resume_years: return 0.25
    return 1.0 if resume_years >= jd_years else 0.0

def match_and_score(jd_skills: List[Dict[str,Any]], resume_skills: List[Dict[str,Any]]):
    rmap = {s["name"].lower(): s for s in resume_skills}
    present_required = present_preferred = 0
    total_required = total_preferred = 0
    level_alignment = years_alignment = 0.0
    matched_rows = []
    missing_required, missing_preferred = [], []
    for js in jd_skills:
        name = js["name"]; req = js.get("requirement","unspecified")
        if req == "required": total_required += 1
        elif req == "preferred": total_preferred += 1
        rs = rmap.get(name.lower())
        if rs:
            if req == "required": present_required += 1
            elif req == "preferred": present_preferred += 1
            lvl_s = _level_score(rs.get("level"), js.get("level"))
            yr_s  = _years_score(rs.get("years"), js.get("years"))
            level_alignment += lvl_s; years_alignment += yr_s
            matched_rows.append({
                "name": name,
                "jd_level": js.get("level"), "resume_level": rs.get("level"),
                "jd_years": js.get("years"), "resume_years": rs.get("years"),
                "requirement": req,
                "confidence_jd": js.get("confidence", 0),
                "confidence_resume": rs.get("score_semantic", 0) or 0,
                "level_score": round(lvl_s,2), "years_score": round(yr_s,2)
            })
        else:
            if req == "required": missing_required.append(name)
            elif req == "preferred": missing_preferred.append(name)
    req_cov = (present_required / total_required) if total_required else 1.0
    pref_cov = (present_preferred / total_preferred) if total_preferred else 1.0
    match_count = max(1, present_required + present_preferred)
    level_align = level_alignment / match_count
    years_align = years_alignment / match_count
    jd_names = [s["name"] for s in jd_skills]
    resume_names = [s["name"] for s in resume_skills]
    groups = rollup_groups(jd_names)
    synergy = 0.0
    for g in groups:
        overlap = len(set(g["skills"]) & set(resume_names))
        if overlap >= 2: synergy += 0.05
    synergy = min(synergy, 0.10)
    score = (
        0.50 * req_cov +
        0.20 * pref_cov +
        0.10 * level_align +
        0.10 * years_align +
        synergy
    )
    score = round(max(0.0, min(1.0, score)) * 100, 1)
    suggestions = list(dict.fromkeys(missing_required + missing_preferred + missing_suggestions(resume_names)))
    return {
        "score": score,
        "coverage": {
            "required": round(req_cov, 2),
            "preferred": round(pref_cov, 2),
            "level_alignment": round(level_align, 2),
            "years_alignment": round(years_align, 2),
            "synergy_bonus": round(synergy, 2)
        },
        "matched": matched_rows,
        "missing": { "required": missing_required, "preferred": missing_preferred },
        "suggestions": suggestions
    }

# =========================
# Flask App
# =========================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/extract")
def extract():
    data = request.get_json(force=True) or {}
    jd = (data.get("job_description") or "").strip()
    if not jd:
        return jsonify({"error": "job_description is required"}), 400
    top_k = int(data.get("top_k", TOP_K))
    use_llm = bool(data.get("use_llm_explanations", True))
    role_hint, industry_hint = detect_role_industry(jd)

    docs = vs.similarity_search_with_score(jd, k=top_k)
    results: Dict[int, Dict[str, Any]] = {}
    for doc, raw_score in docs:
        score = _to_01(float(raw_score))
        if score >= SEM_THRESHOLD:
            meta = doc.metadata
            idx = int(meta["idx"])
            results[idx] = {
                "name": meta.get("name"),
                "category": meta.get("category"),
                "blurb": meta.get("blurb", ""),
                "examples": meta.get("examples", []),
                "score_semantic": round(score, 4),
                "snippets": [],
            }

    alias_idx = alias_hits(jd)
    for idx in alias_idx:
        s = SKILLS[idx]
        entry = results.get(
            idx,
            {
                "name": s["name"],
                "category": s.get("category"),
                "blurb": s.get("blurb", ""),
                "examples": s.get("examples", []),
                "score_semantic": None,
                "snippets": [],
            },
        )
        for needle in [s["name"], *s.get("aliases", [])]:
            entry["snippets"].extend(extract_snippets(jd, needle))
        entry["snippets"] = list(dict.fromkeys(entry["snippets"]))
        results[idx] = entry

    ranked = sorted(results.values(), key=lambda r: (r["score_semantic"] or 0.0), reverse=True)

    # ✅ JD-ONLY FILTER: keep only skills that have at least one snippet
    ranked = [r for r in ranked if r.get("snippets")]

    detected_names: List[str] = []
    rows_for_db: List[Tuple[str, str, float, int, float]] = []

    for r in ranked:
        detected_names.append(r["name"])
        level, years = detect_level_and_years(jd, r["name"])
        reqpref = detect_required_preferred(jd, r["name"])
        if USE_LLM_JUDGE and reqpref == "unspecified":
            reqpref = llm_required_preferred(jd, r["name"])

        base = r["score_semantic"] or 0.50
        if r.get("snippets"): base += 0.07
        confidence = round(min(base, 0.95), 2)

        r["level"] = level
        r["years"] = years if years > 0 else None
        r["requirement"] = reqpref
        r["confidence"] = confidence

        if use_llm:
            try:
                prompt = f"""In 1–2 concise sentences, explain why skill "{r['name']}" matters for THIS job.
Avoid boilerplate; reference responsibilities if helpful.
Role: {role_hint or 'unspecified'} | Industry: {industry_hint or 'unspecified'}

Job description:
{jd}"""
                resp = llm.invoke([HumanMessage(content=prompt)])
                r["why_this_job"] = resp.content.strip()
            except Exception:
                r["why_this_job"] = None

        rows_for_db.append(
            (r["name"], r["level"], r["years"] or 0.0, 1 if reqpref == "required" else 0, r["confidence"])
        )

    groups = rollup_groups(detected_names)
    suggestions = missing_suggestions(detected_names)

    job_id = db_new_job(role_hint, industry_hint)
    db_log(job_id, rows_for_db)

    return jsonify(
        {
            "skills": ranked[:top_k],
            "groups": groups,
            "suggestions": suggestions,
            "context": {"role": role_hint or None, "industry": industry_hint or None},
            "meta": {
                "k": top_k,
                "thresholds": {"semantic": SEM_THRESHOLD, "alias_fuzz": ALIAS_FUZZ_THRESHOLD},
                "models": {"llm": OPENAI_MODEL, "embedding": EMBED_MODEL},
            },
            "job_id": job_id,
        }
    )

@app.post("/resume/score")
def resume_score():
    if "file" not in request.files:
        return jsonify({"error":"Attach a resume file (PDF/DOCX/TXT) as 'file'"}), 400
    job_description = (request.form.get("job_description") or "").strip()
    if not job_description:
        return jsonify({"error":"job_description is required"}), 400
    top_k = int(request.form.get("top_k") or TOP_K)

    try:
        resume_text = read_resume_file(request.files["file"])
    except Exception as e:
        return jsonify({"error": f"Resume read failed: {e}"}), 400

    docs = vs.similarity_search_with_score(job_description, k=top_k)
    results: Dict[int, Dict[str, Any]] = {}
    for doc, raw in docs:
        score = _to_01(float(raw))
        if score >= SEM_THRESHOLD:
            meta = doc.metadata; idx = int(meta["idx"])
            results[idx] = {
                "name": meta.get("name"),
                "category": meta.get("category"),
                "blurb": meta.get("blurb",""),
                "examples": meta.get("examples", []),
                "score_semantic": round(score, 4),
                "snippets": [],
            }
    alias_idx = alias_hits(job_description)
    for idx in alias_idx:
        s = SKILLS[idx]
        entry = results.get(idx, {
            "name": s["name"], "category": s.get("category"),
            "blurb": s.get("blurb",""), "examples": s.get("examples", []),
            "score_semantic": None, "snippets": []
        })
        for needle in [s["name"], *s.get("aliases", [])]:
            entry["snippets"].extend(extract_snippets(job_description, needle))
        entry["snippets"] = list(dict.fromkeys(entry["snippets"]))
        results[idx] = entry

    jd_ranked = sorted(results.values(), key=lambda r:(r["score_semantic"] or 0.0), reverse=True)

    # NOTE: No snippet-based filter here (per your instruction not to change resume comparison)
    for r in jd_ranked:
        lvl, yrs = detect_level_and_years(job_description, r["name"])
        reqpref = detect_required_preferred(job_description, r["name"])
        if USE_LLM_JUDGE and reqpref == "unspecified":
            reqpref = llm_required_preferred(job_description, r["name"])
        r["level"] = lvl
        r["years"] = yrs if yrs > 0 else None
        r["requirement"] = reqpref
        r["confidence"] = r.get("score_semantic", 0)

    resume_skills = extract_skills_from_text(resume_text, top_k=top_k)

    # NOTE: No snippet-based filter here either (keep resume flow unchanged)
    match = match_and_score(jd_ranked[:top_k], resume_skills[:top_k])

    return jsonify({
        "overall_score": match["score"],
        "breakdown": match["coverage"],
        "missing": match["missing"],
        "suggestions": match["suggestions"],
        "matched": match["matched"],
        "resume_skills": resume_skills[:top_k]
    })

@app.get("/stats/cooccurrence")
def cooccurrence():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT job_id, group_concat(skill) FROM job_skills GROUP BY job_id")
    pairs = defaultdict(int)
    for job_id, csv in cur.fetchall():
        if not csv: continue
        skills = sorted(set(csv.split(",")))
        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                pairs[(skills[i], skills[j])] += 1
    con.close()
    top = sorted(
        [{"a": a, "b": b, "count": c} for (a, b), c in pairs.items()],
        key=lambda x: -x["count"],
    )[:50]
    return jsonify({"pairs": top})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
