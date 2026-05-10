import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")

st.title("📄 AI Resume Analyzer")
st.write("Analyze your resume against job descriptions using NLP.")

uploaded_file = st.file_uploader("Upload Your Resume", type="pdf")

job_description = st.text_area("Paste Job Description")

skills = [
    "python",
    "java",
    "sql",
    "machine learning",
    "react",
    "node.js",
    "git",
    "data structures",
    "html",
    "css"
]

if uploaded_file is not None and job_description:

    pdf = PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf.pages:
        resume_text += page.extract_text()

    text = [resume_text, job_description]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text)

    similarity = cosine_similarity(vectors)[0][1]

    score = round(similarity * 100, 2)

    st.subheader(f"ATS Score: {score}%")
    st.progress(int(score))
    if score > 70:
        st.success("Great Match!")
    elif score > 50:
        st.warning("Average Match")
    else:
        st.error("Low Match")

    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    matched_skills = []
    missing_skills = []

    for skill in skills:
        if skill in resume_lower and skill in jd_lower:
            matched_skills.append(skill)

        elif skill in jd_lower and skill not in resume_lower:
            missing_skills.append(skill)

    st.subheader("Matching Skills")
    st.write(matched_skills)

    st.subheader("Missing Skills")
    st.write(missing_skills)