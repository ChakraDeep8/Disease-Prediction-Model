import streamlit as st
from res.multiapp import MultiApp
from Apps import Hypertension_App, Stroke_App, Heart_Disease, Diabetes  # import your app modules here
from PIL import Image
from res import Header as hd

app = MultiApp()
st.set_page_config(
    page_title="Health Track",
    page_icon=Image.open("images/medical-team.png"),
    layout="wide",

)

image = Image.open("images/Health Track.png")
st.sidebar.image(image, use_column_width=True)

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
    image = Image.open('images/med.png')
    st.image(image, use_column_width="auto")
    col1.empty()
with col2:
    col2.empty()
    st.title("Health Track")
    st.markdown("""

    **Disease Detector App** - In the realm of healthcare, predicting diseases before they manifest can be a game-changer. 
    It can lead to early interventions, better management of health conditions, and improved patient outcomes. 
    To this end, we propose the development of a Disease Prediction Model using Machine Learning (ML) techniques.

    This model will analyze various health parameters of an individual and predict the likelihood of them developing a specific disease.

    _The parameters could include_ `age, gender, lifestyle habits, genetic factors, and existing health conditions` _, among others._
    """)
st.write("")
st.write("")
st.write("")
hd.colored_header(
    label="Select your disease",
    color_name="violet-70",
)


# Add all your application here
app.add_app("Heart Disease Detector", Heart_Disease.app)
app.add_app("Hypertension Detector", Hypertension_App.app)
app.add_app("Stroke Detector", Stroke_App.app)
app.add_app("Diabetes Detector", Diabetes.app)
# The main app
app.run()