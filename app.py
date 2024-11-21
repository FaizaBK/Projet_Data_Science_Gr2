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

st.sidebar.markdown(
    """
    <a href="https://github.com/FaizaBK/Projet_Data_Science_Gr2" target="_blank" style="text-decoration: none; margin:20px">
        <button style="background-color: #24292E; color: white; border: none; padding: 10px 20px; font-size: 16px; cursor: pointer; border-radius: 5px; display: flex; align-items: center; ">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="width: 20px; height: 20px; margin-right: 10px;">
            Aller sur GitHub
        </button>
    </a>
    """,
    unsafe_allow_html=True,
)

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
    
    
