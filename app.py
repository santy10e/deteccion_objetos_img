"""Object Detection with Deformable DETR and Label Translation.

Punto de entrada de Streamlit: configura la página, el tema y la navegación.
La lógica de inferencia vive en `core/` y la presentación en `ui/`.

    streamlit run app.py
"""

import streamlit as st

from core.config import APP_NAME, load_settings
from core.pipeline import model_statuses
from ui import detection_page, models_page
from ui.components import sidebar
from ui.theme import inject_theme

st.set_page_config(
    page_title=f"{APP_NAME} · Deformable DETR",
    page_icon=":material/center_focus_strong:",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_theme()

settings = load_settings()
statuses = model_statuses(settings)


def deteccion() -> None:
    detection_page.render(settings, statuses)


def modelos() -> None:
    models_page.render(settings, statuses)


pages = [
    st.Page(deteccion, title="Detección", icon=":material/center_focus_strong:", url_path="deteccion", default=True),
    st.Page(modelos, title="Modelos", icon=":material/memory:", url_path="modelos"),
]
current = st.navigation(pages, position="hidden")
sidebar(settings, statuses, pages)
current.run()
