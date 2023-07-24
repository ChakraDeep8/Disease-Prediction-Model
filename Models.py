import streamlit as st
from multiapp import MultiApp
from Apps import Hypertension_App, Stroke_App, Heart_Disease # import your app modules here

app = MultiApp()
st.set_page_config(
    page_title="Disease Predictor App",
    page_icon="👨‍⚕️",
    layout="wide",
    initial_sidebar_state="collapsed"

)



st.markdown("""
# Disease Detector App
""")

# Add all your application here
app.add_app("Heart Disease Detector", Heart_Disease.app)
app.add_app("Hypertension Detector", Hypertension_App.app)
app.add_app("Stroke Detector", Stroke_App.app)
# The main app
app.run()