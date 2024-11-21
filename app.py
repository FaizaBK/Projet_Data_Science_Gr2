import streamlit as st
from sections.nailsdetection.nails import nail_page
from sections.classification.classification import classification_page
from sections.regression.regression import regression_page



st.set_page_config(
    page_title="ProjectDataScience",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.header("Bienvenue sur notre projet de Data Science")
st.sidebar.link_button("Got to Github", "https://github.com/FaizaBK/Projet_Data_Science_Gr2")

type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Regression", "Classification", "NailsDetection"],
    index=None,
)

if type_data == "Regression":
    regression_page()
elif type_data == "Classification":
    classification_page()
elif type_data == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option dans la sidebard")
    
    
