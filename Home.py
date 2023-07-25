import streamlit as st
from multiapp import MultiApp
from Apps import Hypertension_App, Stroke_App, Heart_Disease  # import your app modules here
from PIL import Image

app = MultiApp()
st.set_page_config(
    page_title="Disease Predictor App",
    page_icon=Image.open("images/medical-team.png"),
    layout="wide",

)


st.markdown(
    """
    <style>
    .markdown-section {
        margin-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap="small")
with col1:
    image = Image.open('images\med.png')
    st.image(image)
    col1.empty()
with col2:
    col2.empty()
    st.markdown("""
    # Disease Detector App

    **In the realm of healthcare, predicting diseases before they manifest can be a game-changer. 
    It can lead to early interventions, better management of health conditions, and improved patient outcomes. 
    To this end, we propose the development of a Disease Prediction Model using Machine Learning (ML) techniques.**

    This model will analyze various health parameters of an individual and predict the likelihood of them developing a specific disease.

    _The parameters could include_ `age, gender, lifestyle habits, genetic factors, and existing health conditions` _, among others._
    """)
st.write("")
st.write("")
st.write("")

# Add all your application here
app.add_app("Heart Disease Detector", Heart_Disease.app)
app.add_app("Hypertension Detector", Hypertension_App.app)
app.add_app("Stroke Detector", Stroke_App.app)
# The main app
app.run()
