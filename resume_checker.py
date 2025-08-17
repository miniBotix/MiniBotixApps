import streamlit as st
import PyPDF2
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# Sample data for training
data = {
    "resume_text": [
        "python sql data analysis power bi statistics",
        "html css javascript react node.js",
        "c c++ arduino raspberry pi microcontroller spi",
        "tensorflow keras python nlp machine learning",
        "aws docker kubernetes devops ci cd terraform"
    ],
    "role": [
        "Data Analyst",
        "Web Developer",
        "Embedded Engineer",
        "AI Engineer",
        "Cloud Engineer"
    ]
}
df = pd.DataFrame(data)

# Encode labels
le = LabelEncoder()
df['encoded_role'] = le.fit_transform(df['role'])

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['resume_text'])
y = df['encoded_role']

model = DecisionTreeClassifier()
model.fit(X, y)

# Role-specific keywords
ROLE_KEYWORDS = {
    "Data Analyst": {"python", "excel", "sql", "power bi", "statistics"},
    "Web Developer": {"html", "css", "javascript", "react", "node.js"},
    "AI Engineer": {"python", "tensorflow", "keras", "machine learning", "nlp"},
    "Embedded Engineer": {"c", "c++", "microcontroller", "arduino", "raspberry pi"},
    "Cloud Engineer": {"aws", "docker", "kubernetes", "terraform", "ci", "cd"}
}

# ---------------- Streamlit UI ----------------
st.title("📄 ML Resume Analyzer")

selected_role = st.selectbox("🎯 Select Target Role", list(ROLE_KEYWORDS.keys()))

uploaded_file = st.file_uploader("📥 Upload Your Resume (PDF)", type="pdf")

if uploaded_file and selected_role:
    # Extract text
    reader = PyPDF2.PdfReader(uploaded_file)
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text().lower()

    # Predict role using ML
    X_input = vectorizer.transform([resume_text])
    predicted_encoded = model.predict(X_input)[0]
    predicted_role = le.inverse_transform([predicted_encoded])[0]

    # Compare prediction with selected role
    confidence = model.predict_proba(X_input)[0][predicted_encoded] * 100

    # Skill match
    resume_words = set(re.sub(r'[^a-zA-Z ]', ' ', resume_text).split())
    target_keywords = ROLE_KEYWORDS[selected_role]
    matched = resume_words.intersection(target_keywords)
    missing = target_keywords - resume_words

    # Match percentage based on skills
    skill_match_percent = (len(matched) / len(target_keywords)) * 100

    # Result
    st.markdown("## 📊 Result Analysis")
    st.write(f"**Selected Role:** {selected_role}")
    st.write(f"**Predicted Role:** {predicted_role}")
    st.write(f"🧠 **Prediction Confidence:** {confidence:.2f}%")
    st.write(f"✅ **Skill Match:** {skill_match_percent:.2f}%")

    if predicted_role.lower() == selected_role.lower() and skill_match_percent >= 60:
        st.success("✅ Your resume is suitable for this role!")
    else:
        st.warning("⚠️ Your resume needs improvement for this role.")

    st.markdown("### ✅ Matched Skills")
    st.write(', '.join(sorted(matched)) if matched else "None")

    st.markdown("### ❌ Missing Skills")
    st.write(', '.join(sorted(missing)) if missing else "None")
