# ***AutoJudge — Predict Programming Problem Difficulty***

## **Project overview**
AutoJudge is a model-based system that predicts the difficulty class (Easy / Medium / Hard) and a numeric difficulty score for programming problems using only textual fields (problem description, input description, output description). The project combines TF–IDF text features with 14 hand-crafted auxiliary features and exposes an interactive Streamlit web interface for real-time inference. 

## **Dataset used**

The dataset contains problem title, description, input_description, output_description, problem_class (Easy/Medium/Hard) and problem_score (numeric). All training and test splits were created from this single dataset. 

## **Approach and models used**

Text pipeline: concatenate description + input_description + output_description → lowercase/clean → TF–IDF (unigrams + bigrams, vocabulary limit).

Auxiliary features: 14 handcrafted features (length, word count, digit count, symbol count, and 10 keyword counts such as graph, dp, tree, math, greedy, recursion, string, array, sort, search), scaled with a StandardScaler. 

***Models:***

Classification: RandomForestClassifier (or other tree-based ensemble selected by validation).

Regression: RandomForestRegressor / GradientBoostingRegressor for numeric difficulty score.

Final inference concatenates TF–IDF features and scaled auxiliary features, then predicts class and score. Artifacts saved as best_clf_model.pkl, best_reg_model.pkl, tfidf_vectorizer.pkl, aux_scaler.pkl, and label_encoder.pkl. 

## **Evaluation metrics**

Classification: Accuracy, Precision, Recall, F1-score, Confusion matrix.

Regression: MAE, MSE, RMSE, R².
(Reported test values in the project: test accuracy ~0.545, test F1 ~0.461, test MAE ~1.687, test RMSE ~2.039 — see report.)

## **Steps to run the project locally**

-Create and activate a virtual environment:

bash

pip install streamlit scikit-learn joblib numpy

-Ensure the trained artifacts are present in the repo root (or update paths in app.py):

best_clf_model.pkl (attached in GitHub Releases Section due to size(>25 MB) issues)

best_reg_model.pkl

tfidf_vectorizer.pkl

aux_scaler.pkl

label_encoder.pkl

-Run the Streamlit app:

'''py

streamlit run app.py

'''

Open the shown local URL in your browser (usually http://localhost:8501) and use the web interface to paste problem text and get predictions

## **Explanation of the web interface**

The Streamlit interface provides three text areas for:

Problem Description (main body)

Input Description

Output Description

**Click Predict to:**

-Build the combined text,

-Transform with the TF–IDF vectorizer,

-Compute the 14 auxiliary features (scaled),

-Concatenate features and run the classification and regression models, and

-**Display Difficulty Class** (Easy/Medium/Hard) and **Difficulty Score** (numeric)

# [**Demo Video Link**](https://drive.google.com/drive/folders/1ofFigP--eAn9MW04qsvpCCrjSBGqsCN3?usp=drive_link)

**Author**

Shrawani Moreshwar Raut
23112093

