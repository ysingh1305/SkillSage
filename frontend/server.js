const express = require("express");
const path = require("path");
const app = express();

app.use(express.json()); // safe: only for application/json
app.use(express.static(path.join(__dirname, "public")));

const FLASK_URL = process.env.FLASK_URL || "http://localhost:5050";

// forward JSON -> /extract
app.post("/extract", async (req, res) => {
  try {
    const r = await fetch(`${FLASK_URL}/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const json = await r.json();
    res.status(r.status).json(json);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Backend not reachable" });
  }
});

// forward multipart -> /resume/score
app.post("/resume/score", async (req, res) => {
  try {
    const headers = { ...req.headers };
    // strip hop-by-hop headers that can confuse the backend
    delete headers["host"];
    const r = await fetch(`${FLASK_URL}/resume/score`, {
      method: "POST",
      headers,
      body: req, // stream the incoming multipart body directly
    });
    const json = await r.json();
    res.status(r.status).json(json);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Backend not reachable" });
  }
});

app.listen(3000, () => console.log("Node UI at http://localhost:3000"));
