from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import streamlit as st
from tempfile import NamedTemporaryFile

# import the inference-sdk
from inference_sdk import InferenceHTTPClient
#  Chargement .env
load_dotenv()

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

def annotate_image(image, predictions, MinConfidence):
    """
    Annoter une image avec des prédictions (bounding boxes et labels).
    """
    draw = ImageDraw.Draw(image)

    for prediction in predictions:
        if prediction["confidence"] >= MinConfidence:
            # Récupérer les coordonnées et la classe
            x, y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
            label = prediction["class"]
            confidence = prediction["confidence"]

            # Calculer les coins du rectangle
            x_min = x - width / 2
            y_min = y - height / 2
            x_max = x + width / 2
            y_max = y + height / 2

            # Dessiner le rectangle et ajouter le label
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, y_min - 10), f"{label} ({confidence:.2f})", fill="red")

    return image

def nail_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans la détection d'ongle")

with st.form("image_form"):

        user_image = st.file_uploader(label='importer une image', type=['png', 'jpg'])
        
        st.form_submit_button("importer l'image")

    # This is outside the form
    if user_image != None:
        
        with open(os.path.join("/tmp", user_image.name), "wb") as f:
            f.write(user_image.read())
        result = CLIENT.infer("/tmp/"+user_image.name, model_id="nailsdiginamic/2")

        with st.form("confidence_form"):
            confidence = st.slider(label='confidence', min_value=0.0, max_value=1.0, value=0.6)
        
            st.form_submit_button("afficher la détection")
        
        st.image(annotate_image(Image.open(user_image), result["predictions"], confidence))
