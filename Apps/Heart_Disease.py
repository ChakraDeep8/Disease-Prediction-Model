import streamlit as st
import pandas as pd
from Classifier_Models import Classifier_model_builder_hypertension as cmb
import pickle
import numpy as np

def app():

    st.write("""
    # Heart Disease Detector
    
    This app predicts whether a person have any heart disease or not

    """)

    st.sidebar.header('User Input Features')
    # st.sidebar.markdown("""
    # [Import input CSV file](https://github.com/ChakraDeep8/Heart-Disease-Detector/tree/master/res)""")


    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def patient_details():
            sex = st.sidebar.selectbox('Sex', ('M', 'F'))
            ChestPainType = st.sidebar.selectbox('Chest Pain Type', ('TA', 'ASY', 'NAP'))
            RestingECG = st.sidebar.selectbox('Resting Electrocardiogram', ('Normal', 'ST', 'LVH'))
            ExerciseAngina = st.sidebar.selectbox('ExerciseAngina', ('Y', 'N'))
            ST_Slope = st.sidebar.selectbox('ST Slope', ('Up', 'Flat', 'Down'))
            Age = st.sidebar.slider('Age', 28, 77)
            RestingBP = st.sidebar.slider('Resting Blood Pressure', 0, 200)
            Cholesterol = st.sidebar.slider('Cholesterol', 0, 603)
            MaxHR = st.sidebar.slider('Maximum Heart Rate', 60, 202)
            Oldpeak = st.sidebar.slider('Old peak', -2, 6)
            FastingBS = st.sidebar.slider('Fasting Blood Sugar', 0, 1)

            data = {'Age': Age,
                    'Sex': sex,
                    'ChestPainType': ChestPainType,
                    'RestingBP': RestingBP,
                    'Cholesterol': Cholesterol,
                    'FastingBS': FastingBS,
                    'RestingECG': RestingECG,
                    'MaxHR': MaxHR,
                    'ExerciseAngina': ExerciseAngina,
                    'Oldpeak': Oldpeak,
                    'ST_Slope': ST_Slope, }

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = patient_details()

    heart_disease_raw = pd.read_csv('res/heart.csv')
    heart = heart_disease_raw.drop(columns=['HeartDisease'])
    df = pd.concat([input_df, heart], axis=0)

    # Encoding of ordinal features
    encode = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
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
    load_clf_NB = pickle.load(open('res/heart_disease_classifier_NB.pkl', 'rb'))
    load_clf_KNN = pickle.load(open('res/heart_disease_classifier_KNN.pkl', 'rb'))
    load_clf_DT = pickle.load(open('res/heart_disease_classifier_DT.pkl', 'rb'))
    load_clf_LR = pickle.load(open('res/heart_disease_classifier_LR.pkl', 'rb'))
    load_clf_RF = pickle.load(open('res/heart_disease_classifier_RF.pkl', 'rb'))

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
            st.write("<p style='font-size:20px;color: orange'><b>You have Heart Disease.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('Naive Bayes Prediction Probability')
        st.write(prediction_proba_NB)
        cmb.plt_NB()

    def KNN():
        st.subheader('K-Nearest Neighbour Prediction')
        knn_prediction = np.array([0, 1])
        if knn_prediction[prediction_KNN] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You have Heart Disease.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('KNN Prediction Probability')
        st.write(prediction_proba_KNN)
        cmb.plt_KNN()

    def DT():
        st.subheader('Decision Tree Prediction')
        DT_prediction = np.array([0, 1])
        if DT_prediction[prediction_DT] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>You have Heart Disease.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('Decision Tree Prediction Probability')
        st.write(prediction_proba_DT)
        cmb.plt_DT()

    def LR():
        st.subheader('Logistic Regression Prediction')
        DT_prediction = np.array([0, 1])
        if DT_prediction[prediction_DT] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>You have Heart Disease.<b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('Logistic Regression Probability')
        st.write(prediction_proba_DT)
        cmb.plt_LR()

    def RF():
        st.subheader('Random Forest Prediction')
        DT_prediction = np.array([0, 1])
        if DT_prediction[prediction_DT] == 1:
            st.write("<p style='font-size:20px; color: orange'><b>You have Heart Disease.</b></p>",
                     unsafe_allow_html=True)
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)
        st.subheader('Random Forest Probability')
        st.write(prediction_proba_DT)
        cmb.plt_LR()

    def predict_best_algorithm():
        NB_prediction = np.array([0, 1])
        knn_prediction = np.array([0, 1])
        DT_prediction = np.array([0, 1])
        LR_prediction = np.array([0, 1])
        RF_prediction = np.array([0, 1])

        if NB_prediction[prediction_NB] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You have Heart Disease. <b></p>",
                     unsafe_allow_html=True)
            st.write("<p style='font-size:24px;'>Best Algorithm: Naive Bayes</p>", unsafe_allow_html=True)
            cmb.plt_NB()
        elif knn_prediction[prediction_KNN] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You have Heart Disease.</b></p>",
                     unsafe_allow_html=True)
            st.write("<p style='font-size:24px;'>Best Algorithm: K-Nearest Neighbour</p>", unsafe_allow_html=True)
            cmb.plt_KNN()
        elif DT_prediction[prediction_DT] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You have Heart Disease.</b></p>",
                     unsafe_allow_html=True)
            st.write("<p style='font-size:24px;'>Best Algorithm: Decision Tree</p>", unsafe_allow_html=True)
            cmb.plt_DT()
        elif LR_prediction[prediction_LR] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You have Heart Disease.</b></p>",
                     unsafe_allow_html=True)
            st.write("<p style='font-size:24px;'>Best Algorithm: Logistic Regression</p>", unsafe_allow_html=True)
            cmb.plt_LR()
        elif RF_prediction[prediction_RF] == 1:
            st.write("<p style='font-size:20px;color: orange'><b>You have Heart Disease.</b></p>",
                     unsafe_allow_html=True)
            st.write("<p style='font-size:24px;'>Best Algorithm: Random Forest</p>", unsafe_allow_html=True)
            cmb.plt_RF()
        else:
            st.write("<p style='font-size:20px;color: green'><b>You are fine.</b></p>", unsafe_allow_html=True)

    # Displays the user input features
    st.subheader('Patient Report')
    st.dataframe(input_df)
    predict_best_algorithm()
