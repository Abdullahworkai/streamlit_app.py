### README.md (detailed)

```markdown
# AI Eval Cloud

Streamlit app to evaluate student writing PDFs using CrewAI. Extracts cadet metadata from PDFs, injects KLPs, runs 5 agents, and exports Excel.

---

## Prerequisites
- Python 3.9–3.11
- An OpenAI API key with access to GPT-4o mini
- Files in this repo:
  - `streamlit_app.py`
  - `requirements.txt`
  - `.gitignore`
  - `.env.example`
  - `KLPs.json` (optional; you can upload in the app)

---

## Local Setup

1) Clone and enter the project
```bash
git clone <your-repo-url> ai-eval-cloud
cd ai-eval-cloud
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Set your OpenAI key (temporary shell env)
```bash
export OPENAI_API_KEY="your-key"
```
Or create a local `.env` (use `.env.example` as a template) and add:
```
OPENAI_API_KEY=your-key
```

4) Run the app
```bash
streamlit run streamlit_app.py
```

5) In the app:
- Sidebar → upload `KLPs.json` (or keep your repo copy)
- Upload your student PDF
- Click “Process PDF”
- After results, use the feedback slider and comments
- Download the Excel report

---

## Deploy on Streamlit Cloud (step-by-step)

1) Push this repo to GitHub (public or private).

2) Go to Streamlit Cloud → “New app” → connect your GitHub repo.

3) App configuration:
- App file: `streamlit_app.py`
- Python version: 3.10 or 3.11 (default is fine)
- Requirements file: `requirements.txt`

4) Add Secrets:
- In the Streamlit Cloud app page → “Settings” → “Secrets” → paste:
```
OPENAI_API_KEY = "your-key"
```

5) Deploy:
- Click “Deploy”
- If you included `KLPs.json` in the repo, it will load from the sidebar upload or your local copy. If not, just upload `KLPs.json` via the sidebar when the app is running.

6) Common Cloud fixes:
- ModuleNotFoundError: ensure `requirements.txt` is present in the repo root and includes:
  - streamlit
  - crewai
  - langchain-openai
  - langchain
  - pypdf2
  - pandas
  - openpyxl
  - openai
- Wrong entrypoint: make sure the app file is `streamlit_app.py`.
- API key issues: confirm the key exists in Secrets and the name is exactly `OPENAI_API_KEY`.

---

## Repository Layout

```
.
├─ streamlit_app.py          # Main Streamlit app (reads key from Secrets/env)
├─ requirements.txt          # Dependencies
├─ KLPs.json                 # Optional KLPs (can also upload at runtime)
├─ .env.example              # Local dev template (do NOT commit .env)
├─ .gitignore                # Ignores runtime & local environment files
└─ README.md                 # This guide
```

`.gitignore` covers:
```
.streamlit/
.env
long_term_memory.json
crew.log
__pycache__/
*.pyc
.DS_Store
```

---

## Usage Notes

- PDF structure: the parser expects patterns like:
  - `For: 12345`
  - `Book 18` (→ becomes `Book_18`)
  - “Essay Question” section
  - Student response before “Instructor Comments:”
- The app removes artifacts like “Section 1”.
- Five agents (manager, grammar/vocab, task achievement, range/accuracy, QA).
- A 45s timeout ensures the app doesn’t hang.
- Feedback is saved to `long_term_memory.json` (ignored by git).

---

## Troubleshooting

- No responses found:
  - Check the PDF text contains “For: <digits>” and “Book <number>”.
  - Ensure the question lines match “Write a paragraph between …” and optional “- Write about …”.
- Timeouts:
  - Try again; ensure stable network.
  - Reduce the number of pages/size of the PDF.
- KLPs missing:
  - Upload `KLPs.json` via sidebar or keep it in the repo and load it through the upload.

---

## Security

- Never commit your real `.env`.
- Streamlit Cloud Secrets store your `OPENAI_API_KEY`.
- This repo includes `.env.example` for local development convenience.

---

## Commands Recap

Local:
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
streamlit run streamlit_app.py
```

Cloud:
- App file: `streamlit_app.py`
- Requirements: `requirements.txt`
- Secrets: `OPENAI_API_KEY = "your-key"`
```

```
