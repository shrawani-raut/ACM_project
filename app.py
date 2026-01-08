import streamlit as st
import joblib
import numpy as np
import re

clf_model = joblib.load("best_clf_model.pkl")
reg_model = joblib.load("best_reg_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
aux_scaler = joblib.load("aux_scaler.pkl")   # only if used

CLASS_MAP = {
    0: "easy",
    1: "medium",
    2: "hard"
}

def build_aux_features(text):
    text = text.lower()

    # ---- Basic text stats  ----
    length = len(text)
    word_count = len(text.split())
    digit_count = sum(c.isdigit() for c in text)
    symbol_count = len(re.findall(r"[+\-*/=<>]", text))

    # ---- Keyword-based features ----
    keyword_graph = text.count("graph")
    keyword_dp = text.count("dp")
    keyword_tree = text.count("tree")
    keyword_math = text.count("math")
    keyword_greedy = text.count("greedy")
    keyword_recursion = text.count("recursion")
    keyword_string = text.count("string")
    keyword_array = text.count("array")
    keyword_sort = text.count("sort")
    keyword_search = text.count("search")

    # ---- Combine ALL 14 in correct shape ----
    aux = np.array([[
        length,
        word_count,
        digit_count,
        symbol_count,
        keyword_graph,
        keyword_dp,
        keyword_tree,
        keyword_math,
        keyword_greedy,
        keyword_recursion,
        keyword_string,
        keyword_array,
        keyword_sort,
        keyword_search
    ]])

    return aux
# ---------------- UI ----------------
st.set_page_config(page_title="AutoJudge – Problem Difficulty Predictor")

st.title(" AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.write("Paste the problem details below:")

problem_desc = st.text_area(" Problem Description", height=150)
input_desc = st.text_area(" Input Description", height=100)
output_desc = st.text_area(" Output Description", height=100)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    text = problem_desc + " " + input_desc + " " + output_desc

    X_tfidf = tfidf.transform([text]).toarray()
    X_aux = build_aux_features(text)
    X_aux_scaled = aux_scaler.transform(X_aux)

    X = np.hstack([X_tfidf, X_aux_scaled])

    raw_pred = clf_model.predict(X)[0]   # e.g. "easy"
    pred_class = raw_pred.strip().capitalize()



    pred_score = reg_model.predict(X)[0]

    st.success(f"Difficulty Class: {pred_class}")
    st.success(f"Difficulty Score: {pred_score:.2f}")

       