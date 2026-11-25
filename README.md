# MedTranslate

MedTranslate provides educational explanations of printed prescriptions and laboratory reports. It uses a Flask web UI, Gemini structured extraction, and DailyMed medicine-label retrieval for prescription context.

## Run on Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your private Google Gemini API key:

```dotenv
GEMINI_API_KEY=your_private_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

Then run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000.

## AI pipeline

1. `backend/document_processor.py` validates PDF/JPG/JPEG/PNG uploads (10 MB maximum), extracts selectable PDF text, and prepares scanned PDFs/images for multimodal processing.
2. `backend/ai_processor.py` uses Gemini to classify the document as `prescription`, `lab_report`, or `unknown` and extract only explicitly written fields.
3. For prescription medicines, `backend/medical_retriever.py` retrieves public medicine-label information from DailyMed.
4. `backend/explanation_generator.py` creates patient-friendly medicine context while keeping the prescription's extracted dosage, frequency, timing and duration unchanged.
5. The dashboard renders extracted fields, medicine explanations and source links. It does not invent lab-range status or questions when the backend has not supplied them.

## API

- `GET /api/health` — reports service health and whether a Gemini key is configured (never returns the key).
- `POST /api/analyze` — multipart form fields: `file` and optional `language`.

Supported uploads: PDF, PNG, JPG, JPEG. Uploads are processed temporarily and deleted after the request.

## Pages

| Route | What |
|---|---|
| `/` | Landing page |
| `/login` | Visual-only sign-in demonstration (no accounts or authentication) |
| `/dashboard` | Document upload and AI analysis |

## Safety and privacy

- Educational information only; not diagnosis, treatment or prescribing.
- AI extracts only text explicitly present in the document. Verify all fields against the original.
- Lab values are not classified as normal or abnormal by this backend.
- Medicine context is educational and linked to DailyMed when a source is available.
- Never commit `.env`, API keys, or patient documents. Use synthetic documents for development.
- The Flask development server is for local testing, not public production deployment.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```
