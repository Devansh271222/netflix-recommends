# Netflix Recommender

A content-based Netflix recommendation system built with Python, TF-IDF, and cosine similarity.

## Features

- Uses Netflix titles metadata
- Genre weighted 3×
- Director weighted 2×
- Cast weighted 1×
- Country weighted 1×
- Case-insensitive title selection
- Uses `NearestNeighbors(metric="cosine")` instead of storing a full dense similarity matrix, making deployment more memory-efficient

## Files

- `app.py` — Streamlit application
- `netflix_titles.csv` — Netflix dataset
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload these files to a GitHub repository and deploy the repository using Streamlit Community Cloud.
