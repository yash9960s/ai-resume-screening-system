# AI Resume Screening System

## Overview

This project uses Natural Language Processing (NLP) techniques to compare resumes with a job description and rank candidates based on similarity.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## Features

- Resume ranking
- Job description matching
- NLP text processing
- Similarity scoring
- Recommendation status

## How It Works

1. Load resumes from CSV.
2. Accept job description.
3. Convert text into TF-IDF vectors.
4. Calculate cosine similarity.
5. Rank resumes by match score.

## Run

```bash
pip install -r requirements.txt
python resume_screening.py
```