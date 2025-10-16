# SkillSage Lite

SkillSage Lite is a web application that extracts skills from job descriptions and compares them with candidate resumes. It uses natural language processing and semantic search to identify key technical skills, classify them, and generate a match score.

## Technologies Used
- **Backend**: Flask, LangChain, FAISS, RapidFuzz, SQLite
- **NLP/ML**: OpenAI Embeddings, GPT models
- **Parsing**: pdfplumber, docx2txt
- **Frontend**: Node.js

## What It Does
- Extracts technical skills from unstructured job descriptions
- Detects whether skills are required or preferred
- Estimates skill level and years of experience mentioned
- Suggests commonly associated skills that may be missing
- Compares resumes to job descriptions and generates a match score with a breakdown
- Stores and analyzes skill co-occurrence patterns

## Key Features Seen by Users
- **Skill Extraction**: Returns only skills actually mentioned in the JD with relevant snippets.
- **Resume Scoring**: Compares skills between a job description and a resume.
- **Confidence Scoring**: Displays confidence levels for detected skills.
- **Suggestions**: Recommends related skills to improve match.
- **Role & Industry Hints**: Infers job role and industry from the description.
- **Performance**: Processes job descriptions in seconds with lightweight architecture.
