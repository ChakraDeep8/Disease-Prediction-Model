import streamlit as st
import pandas as pd
import Classifier_model_builder_hypertension as cmb
import pickle
import numpy as np

st.set_page_config(
    page_title="Stroke Detector",
    page_icon="🩸"
)

st.write("""
# Hypertension Blood Pressure Detector

This app predicts whether a person have any hypertension blood pressure or not

""")

st.sidebar.header('User Input Features')
# st.sidebar.markdown("""
# [Import input CSV file](https://github.com/ChakraDeep8/Heart-Disease-Detector/tree/master/res)""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def patient_details():
        age = st.sidebar.slider('Age', 0, 98)
        sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
        hypertension = st.sidebar.selectbox('Hypertension',['Yes', 'No'])
        heart_disease = st.sidebar.selectbox('Marraige Status', ['Yes', 'No'])
        serum_cholesterol = st.sidebar.slider('Serum Cholesterol', 126, 564)
        fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar',
                                          ['Yes', 'No'])  # if the patient's fasting blood sugar > 120 mg/dl
        resting_ecg = st.sidebar.selectbox('Resting ECG',
                                           ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
        max_hr = st.sidebar.slider('Max Heart Rate', 71, 202)
        exercise_angina = st.sidebar.selectbox('Exercise-Induced Angina', ['Yes', 'No'])
        oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2)
        st_slope = st.sidebar.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])
        major_vessels = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4)
        thalassemia = st.sidebar.slider('Thalassemia', 0, 3)

        data = {'age': age,
                'sex': sex,
                'cp': hypertension,
                'trestbps': resting_bp,
                'chol': serum_cholesterol,
                'fbs': fasting_bs,
                'restecg': resting_ecg,
                'thalach': max_hr,
                'exang': exercise_angina,
                'oldpeak': oldpeak,
                'slope': st_slope,
                'ca': major_vessels,
                'thal': thalassemia, }

        features = pd.DataFrame(data, index=[0])
        return features


    input_df = patient_details()

hypertension_disease_raw = pd.read_csv('res/hypertension_data.csv')
hypertension = hypertension_disease_raw.drop(columns=['target'])
df = pd.concat([input_df, hypertension], axis=0)

# Encoding of ordinal features
encode = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # Selects only the first row (the user input data)
df.loc[:, ~df.columns.duplicated()]

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    df = df.loc[:, ~df.columns.duplicated()]
    st.write(df)

# Load the classification models
load_clf_NB = pickle.load(open('res/hypertension_disease_classifier_NB.pkl', 'rb'))
load_clf_KNN = pickle.load(open('res/hypertension_disease_classifier_KNN.pkl', 'rb'))
load_clf_DT = pickle.load(open('res/hypertension_disease_classifier_DT.pkl', 'rb'))
load_clf_LR = pickle.load(open('res/hypertension_disease_classifier_LR.pkl', 'rb'))
load_clf_RF = pickle.load(open('res/hypertension_disease_classifier_RF.pkl', 'rb'))

# Apply models to make predictions
prediction_NB = load_clf_NB.predict(df)
prediction_proba_NB = load_clf_NB.predict_proba(df)
prediction_KNN = load_clf_KNN.predict(df)
prediction_proba_KNN = load_clf_KNN.predict_proba(df)
prediction_DT = load_clf_DT.predict(df)
prediction_proba_DT = load_clf_DT.predict_proba(df)
prediction_LR = load_clf_LR.predict(df)
prediction_proba_LR = load_clf_LR.predict_proba(df)
prediction_RF = load_clf_RF.predict(df)
prediction_proba_RF = load_clf_RF.predict_proba(df)


def NB():
    st.subheader('Naive Bayes Prediction')
    NB_prediction = np.array([0, 1])
    if NB_prediction[prediction_NB] == 1:
        st.write("<p style='font-size:20px;color: orange'></p>", unsafe_allow_html=True)
    else:
        st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
    st.subheader('Naive Bayes Prediction Probability')
    st.write(prediction_proba_NB)
    cmb.plt_NB()


def KNN():
    st.subheader('K-Nearest Neighbour Prediction')
    knn_prediction = np.array([0, 1])
    if knn_prediction[prediction_KNN] == 1:
        st.write("<p style='font-size:20px;color: orange'><b>Heart Disease Detected.</b></p>", unsafe_allow_html=True)
    else:
        st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
    st.subheader('KNN Prediction Probability')
    st.write(prediction_proba_KNN)
    cmb.plt_KNN()


def DT():
    st.subheader('Decision Tree Prediction')
    DT_prediction = np.array([0, 1])
    if DT_prediction[prediction_DT] == 1:
        st.write("<p style='font-size:20px; color: orange'><b>Heart Disease Detected.</b></p>", unsafe_allow_html=True)
    else:
        st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
    st.subheader('Decision Tree Prediction Probability')
    st.write(prediction_proba_DT)
    cmb.plt_DT()


def LR():
    st.subheader('Logistic Regression Prediction')
    DT_prediction = np.array([0, 1])
    if DT_prediction[prediction_DT] == 1:
        st.write("<p style='font-size:20px; color: orange'><b>You have hypertension.</b></p>", unsafe_allow_html=True)
    else:
        st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
    st.subheader('Logistic Regression Probability')
    st.write(prediction_proba_DT)
    cmb.plt_LR()


def RF():
    st.subheader('Random Forest Prediction')
    DT_prediction = np.array([0, 1])
    if DT_prediction[prediction_DT] == 1:
        st.write("<p style='font-size:20px; color: orange'><b>You have hypertension.</b></p>", unsafe_allow_html=True)
    else:
        st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
    st.subheader('Random Forest Probability')
    st.write(prediction_proba_DT)
    cmb.plt_LR()


def select_best_algorithm():
    # Create a dictionary to store the accuracies
    accuracies = {
        'Naive Bayes': cmb.nb_accuracy,
        'KNN': cmb.knn_accuracy,
        'Decision Tree': cmb.dt_accuracy,
        'Logistic Regression': cmb.lr_accuracy,
        'Random Forest': cmb.rf_accuracy

    }

    # Find the algorithm with the highest accuracy
    best_algorithm = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_algorithm]

    # Display the results
    st.write("<p style='font-size:24px;'>Best Algorithm: {}</p>".format(best_algorithm), unsafe_allow_html=True)


def predict_best_algorithm():
    NB_prediction = np.array([0, 1])
    knn_prediction = np.array([0, 1])
    DT_prediction = np.array([0, 1])
    LR_prediction = np.array([0 , 1])
    RF_prediction = np.array([0 , 1])

    if NB_prediction[prediction_NB] == 1:
        st.write("<p style='font-size:20px;color: orange'><b>You have hypertension. <b></p>", unsafe_allow_html=True)
        cmb.plt_NB()
    elif knn_prediction[prediction_KNN] == 1:
        st.write("<p style='font-size:20px;color: orange'><b>You have hypertension.</b></p>", unsafe_allow_html=True)
        cmb.plt_KNN()
    elif DT_prediction[prediction_DT] == 1:
        st.write("<p style='font-size:20px;color: orange'><b>You have hypertension.</b></p>", unsafe_allow_html=True)
        cmb.plt_DT()
    elif LR_prediction[prediction_LR] == 1:
        st.write("<p style='font-size:20px;color: orange'><b>You have hypertension.</b></p>", unsafe_allow_html=True)
        cmb.plt_DT()
    elif RF_prediction[prediction_RF] == 1:
        st.write("<p style='font-size:20px;color: orange'><b>You have hypertension.</b></p>", unsafe_allow_html=True)
        cmb.plt_DT()
    else:
        st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)


# Displays the user input features
st.subheader('Patient Report')
select_best_algorithm()
predict_best_algorithm()
