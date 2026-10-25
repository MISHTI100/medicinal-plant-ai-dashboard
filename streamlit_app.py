
import os
import json
import streamlit as st
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from groq import Groq

st.set_page_config(
    page_title="Medicinal Plant AI",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Medicinal Plant AI Agent")
st.write(
    "Identify medicinal plants using image classification "
    "and explore information through a Groq-powered AI assistant."
)


import os
import gdown

MODEL_PATH = "/tmp/hog_models.joblib"
DRIVE_FILE_ID = "1_cuDRZG-YuXrLd5Xavb86DEljk4ZwdSo"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        id=DRIVE_FILE_ID,
        output=MODEL_PATH,
        quiet=False
    )

plant_database = {
    "tulsi": {
        "scientific_name": "Ocimum tenuiflorum",
        "traditional_uses": [
            "Traditionally used in herbal preparations"
        ]
    },
    "neem": {
        "scientific_name": "Azadirachta indica",
        "traditional_uses": [
            "Traditionally used in herbal preparations"
        ]
    },
    "aloe vera": {
        "scientific_name": "Aloe vera",
        "traditional_uses": [
            "Used in some traditional and topical preparations"
        ]
    }
}

@st.cache_resource
def load_model():
    models = joblib.load(MODEL_PATH)
    return models["Linear SVM (HOG)"]

if not os.path.exists(MODEL_PATH):
    st.warning(
        "Model file is missing. Upload hog_models.joblib "
        "to the project root before identifying plants."
    )
    st.stop()

try:
    svm_model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

uploaded_image = st.file_uploader(
    "Upload a medicinal plant leaf image",
    type=["jpg", "jpeg", "png"]
)

question = st.text_input(
    "Ask a question about the predicted plant",
    placeholder="What is its scientific name?"
)

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded leaf", width=350)

    if st.button("Identify Plant", type="primary"):
        resized = image.resize((64, 64))
        features = hog(
            np.array(resized),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=-1
        )

        predicted = str(
            svm_model.predict(features.reshape(1, -1))[0]
        )
        plant_name = predicted.lower().strip().split(".")[0].strip()

        if plant_name in ("aloe", "aloevera"):
            plant_name = "aloe vera"

        st.session_state["predicted"] = predicted
        st.session_state["plant_name"] = plant_name

if "predicted" in st.session_state:
    predicted = st.session_state["predicted"]
    plant_name = st.session_state["plant_name"]

    st.subheader("🔍 Prediction")
    st.success(f"Predicted class: {predicted}")

    info = plant_database.get(plant_name)

    if info:
        st.subheader("🌱 Plant Information")
        st.write("**Scientific name:**", info["scientific_name"])
        st.write("**Recorded traditional uses:**")
        for use in info["traditional_uses"]:
            st.write(f"- {use}")

        if st.button("Ask Groq AI Agent"):
            api_key = st.secrets.get(
                "GROQ_API_KEY",
                os.getenv("GROQ_API_KEY", "")
            )

            if not api_key:
                st.error(
                    "Add GROQ_API_KEY in Streamlit secrets "
                    "before using the AI agent."
                )
            else:
                try:
                    client = Groq(api_key=api_key)
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer only from the supplied plant "
                                    "record. Do not invent medicinal uses "
                                    "or claim a plant cures diseases."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Question: {question or 'Describe this plant.'}\n"
                                    f"Record: {json.dumps(info)}"
                                )
                            }
                        ],
                        temperature=0
                    )
                    st.write(response.choices[0].message.content)
                except Exception as error:
                    st.error(f"Groq request failed: {error}")
    else:
        st.info(
            f"No database entry exists for '{plant_name}'. "
            "The model prediction is not a verified medicinal identification."
        )
