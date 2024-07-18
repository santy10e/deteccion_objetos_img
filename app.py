import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# API URLs and headers
DETR_API_URL = "https://api-inference.huggingface.co/models/SenseTime/deformable-detr"
TRANSLATION_API_URL = (
    "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-es"
)
headers = {"Authorization": "Bearer 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"}

# Function to query the object detection model
def query_detr(image):
    response = requests.post(DETR_API_URL, headers=headers, data=image)
    try:
        return response.json()
    except ValueError:
        st.error("Error decoding JSON response from DETR API")
        return []

# Function to translate labels
def translate_label(text):
    payload = {"inputs": text}
    response = requests.post(TRANSLATION_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()[0].get("translation_text", text)
        except (KeyError, IndexError, TypeError):
            st.error("Error processing translation API response")
            return text
    else:
        st.error("Translation API request failed")
        return text

# Streamlit App
st.set_page_config(page_title="Object Detection", page_icon="🔍", layout="centered")
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f0f0;
            padding: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stFileUploader>div>div {
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Object Detection with Deformable DETR and Label Translation")
st.markdown(
    "Esta aplicación utiliza el modelo **Deformable DETR** para la detección de objetos y el modelo **Helsinki-NLP** para traducir las etiquetas de los objetos detectados del inglés al español."
)
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Imagen subida", use_column_width=True)

    st.write("Clasificando...")
    detections = query_detr(image_bytes)

    if detections:
        # Display the image with bounding boxes
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')  # Hide axes

        # Add bounding boxes
        for detection in detections:
            if (
                isinstance(detection, dict)
                and "box" in detection
                and "label" in detection
                and "score" in detection
            ):
                box = detection["box"]
                label = detection["label"]
                translated_label = translate_label(label)
                score = detection["score"]

                # Create a Rectangle patch
                rect = patches.Rectangle(
                    (box["xmin"], box["ymin"]),
                    box["xmax"] - box["xmin"],
                    box["ymax"] - box["ymin"],
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )

                # Add the patch to the Axes
                ax.add_patch(rect)

                # Add label and score with background for readability
                plt.text(
                    box["xmin"],
                    box["ymin"] - 10,
                    f"{translated_label}: {score:.2f}",
                    color="red",
                    fontsize=12,
                    weight="bold",
                    bbox=dict(facecolor="white", alpha=0.5),
                )

        # Display the result
        st.pyplot(fig)
    else:
        st.write("No se detectaron objetos.")
