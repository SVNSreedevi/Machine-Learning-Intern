import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read skills list
skills_list = []
with open("skills.txt", "r") as f:
    skills_list = [s.strip().lower() for s in f.readlines()]

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Extract skills present in resume
def extract_skills(resume_text):
    found_skills = []
    for skill in skills_list:
        if skill in resume_text.lower():
            found_skills.append(skill)
    return list(set(found_skills))

# Match resume with job description (Similarity)
def calculate_similarity(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

# Main Program
while True:
    print("\n==== Resume Analyzer ====")

    resume = input("\nPaste your resume text:\n")
    jd = input("\nPaste Job Description:\n")

    resume_clean = clean_text(resume)
    jd_clean = clean_text(jd)

    # Skill extraction
    skills_found = extract_skills(resume)

    # Similarity score
    resume_score = calculate_similarity(resume_clean, jd_clean)

    # Missing skills
    missing = list(set(skills_list) - set(skills_found))

    print("\n========= RESULT =========")
    print(f"Match Score: {resume_score}%")

    print("\nSkills Found in Resume:")
    for s in skills_found:
        print(" -", s)

    print("\nMissing Skills:")
    for m in missing:
        print(" -", m)

    print("\n==========================")

    again = input("\nAnalyze another resume? (yes/no): ")
    if again.lower() != "yes":
        break
