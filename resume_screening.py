import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load resume dataset
df = pd.read_csv("resumes.csv")

print("\n=== AI Resume Screening System ===\n")

job_description = input("Enter Job Description:\n")

# Convert resumes to list
documents = df["Resume"].tolist()

# Add job description
documents.append(job_description)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(documents)

# Separate vectors
job_vector = tfidf_matrix[-1]

resume_vectors = tfidf_matrix[:-1]

# Calculate similarity
scores = cosine_similarity(job_vector, resume_vectors)

# Store scores
df["Match Score"] = scores[0]

# Sort descending
result = df.sort_values(
    by="Match Score",
    ascending=False
)

print("\nTop Matching Resumes:\n")

for index, row in result.iterrows():

    if row["Match Score"] > 0.5:
        status = "Highly Recommended"

    elif row["Match Score"] > 0.2:
        status = "Recommended"

    else:
        status = "Low Match"

    print(f"Category: {row['Category']}")
    print(f"Score: {round(row['Match Score']*100,2)}%")
    print(f"Status: {status}")
    print("-"*40)