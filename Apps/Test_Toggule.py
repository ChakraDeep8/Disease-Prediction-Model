import streamlit as st
from streamlit_toggle import st_toggle_switch
def app():
        st.write("## Toggle Switch")
        st_toggle_switch(
                label="Enable Setting?",
                key="switch_1",
                default_value=False,
                label_after=False,
                inactive_color="#D3D3D3",  # optional
                active_color="#11567f",  # optional
                track_color="#29B5E8",  # optional
        )