# AI-Based Early Detection of Online Radicalization using Deep Learning and Natural Language Processing

This project is a production-style academic prototype that detects early signs of online radicalization from textual content. It combines multilingual-friendly NLP preprocessing, a trained multi-class classifier, emotion detection, explainable AI, risk scoring, SQLite persistence, and a polished FastAPI dashboard.

## Project Highlights

- Multi-class prediction: `Normal`, `Suspicious`, `Radical`
- Manual text input and batch analysis
- NLP preprocessing with cleaning, tokenization, stopword removal, and optional lemmatization/stemming
- Baseline feature extraction with word-level and character-level TF-IDF
- Transformer-ready backend hook for later mBERT or BERT upgrades
- Emotion detection for `anger`, `fear`, and `hate`
- Probability outputs and confidence scores
- Explainable AI with `LIME`
- Risk score from `0-100` with `Low Risk`, `Medium Risk`, and `High Risk`
- SQLite storage for anonymized analysis history
- FastAPI web UI with dashboard, charts, alerts, metrics, and history
- Retraining endpoint for continuous demo refresh

## Tech Stack

- Backend: `FastAPI`, `Jinja2`, `SQLAlchemy`
- NLP / ML: `scikit-learn`, `nltk`, `lime`, `transformers`
- Data: `pandas`, `numpy`, `SQLite`
- Frontend: custom HTML, CSS, JavaScript

## Folder Structure

```text
project/
|
|-- data/
|   |-- combined_training_dataset.csv
|   |-- extremism_data_final.csv
|   `-- sample_radicalization_dataset.csv
|-- models/
|-- static/
|   |-- css/
|   |   `-- styles.css
|   `-- js/
|       `-- app.js
|-- templates/
|   `-- index.html
|-- app.py
|-- database.py
|-- model.py
|-- predict.py
|-- preprocessing.py
|-- schemas.py
|-- train.py
|-- requirements.txt
`-- README.md
```

## How It Works

1. The user enters text or a batch of texts in the dashboard.
2. The preprocessing pipeline cleans the text, removes stopwords, tokenizes, anonymizes sensitive patterns, and detects language.
3. A trained multi-class logistic regression model built on TF-IDF features predicts `Normal`, `Suspicious`, or `Radical`.
4. Emotion signals are estimated using anger, fear, and hate lexicons.
5. A risk score is computed from the class probabilities and emotion profile.
6. `LIME` highlights influential words for interpretability.
7. Results are stored in SQLite and reflected in the dashboard history and alert stream.

## Dataset Strategy

The demo trains from `data/combined_training_dataset.csv`, which maps `data/extremism_data_final.csv` into the project schema and keeps the curated academic sample data for the `Suspicious` class. The training CSV uses these columns:

- `text`
- `label`
- `source`
- `language`

Labeling strategy:

- `Normal`: neutral, civic, educational, or routine conversation
- `Suspicious`: grievance-driven, polarizing, coded, or escalating rhetoric
- `Radical`: explicit support for violent retaliation, dehumanization, or aggressive extremist framing

Recommended real-world expansion sources:

- Kaggle extremist speech datasets
- Twitter discussion datasets
- Reddit comment corpora
- Expert-reviewed moderation datasets

## Run The Project

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the academic demo model:

```bash
python train.py
```

3. Start the FastAPI app:

```bash
uvicorn app:app --reload
```

4. Open the dashboard:

```text
http://127.0.0.1:8000
```

## API Endpoints

- `GET /` : dashboard UI
- `GET /health` : health check
- `POST /api/analyze` : analyze a single text
- `POST /api/analyze-batch` : analyze multiple texts
- `GET /api/history` : fetch recent analysis history
- `GET /api/dashboard` : fetch dashboard summary and metrics
- `POST /api/retrain` : retrain the demo model
- `GET /api/alerts/stream` : live alert event stream

## Evaluation Metrics

The dashboard surfaces:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Training artifacts are stored under `models/`.

## Ethical Safeguards

- The system stores anonymized previews instead of raw user text.
- Predictions should support human moderation, not replace it.
- The sample dataset is academic and small, so outputs can be biased or overconfident.
- Real deployment should include human review, bias audits, and legal-compliance checks.

## Transformer Upgrade Path

The current demo runs locally with the TF-IDF classifier by default. `model.py` includes a transformer-ready hook so you can later enable:

- `mBERT` for Hindi and multilingual expansion
- `BERT` or `RoBERTa` fine-tuning with PyTorch
- Embedding fusion with the TF-IDF baseline

To enable live transformer embeddings, install `torch` and provide pretrained model weights.
