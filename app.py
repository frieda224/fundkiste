import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from datetime import date

# ----------------------------
# Konfiguration
# ----------------------------
MODEL_PATH = "model/keras_model.h5"
LABELS_PATH = "model/labels.txt"
UPLOAD_DIR = "uploads"
DATA_FILE = "data.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Modell & Labels laden
# ----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.asarray(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])

    return labels[index], confidence

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Digitales Fundbüro", page_icon="📦")
st.title("📦 Digitales Fundbüro (Schule)")

tab1, tab2 = st.tabs(["📸 Fund erfassen", "🔍 Fund suchen"])

# ==================================================
# TAB 1: Fund erfassen
# ==================================================
with tab1:
    st.header("Gefundenen Gegenstand erfassen")

    eingabe_art = st.radio(
        "Bildquelle auswählen:",
        ["📷 Kamera verwenden", "📁 Datei hochladen"]
    )

    image = None

    if eingabe_art == "📷 Kamera verwenden":
        camera_image = st.camera_input("Foto aufnehmen")
        if camera_image:
            image = Image.open(camera_image)
            st.image(image, caption="Aufgenommenes Foto", use_column_width=True)

    if eingabe_art == "📁 Datei hochladen":
        uploaded_file = st.file_uploader(
            "Bilddatei hochladen",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if image:
        beschreibung = st.text_input("Kurze Beschreibung")
        fundort = st.text_input("Fundort")
        funddatum = st.date_input("Funddatum", value=date.today())

        label, confidence = predict_image(image)
        st.info(f"🤖 KI-Erkennung: **{label}** ({confidence:.2%})")

        if st.button("Fund speichern"):
            data = load_data()
            filename = f"{len(data)}.jpg"
            image_path = os.path.join(UPLOAD_DIR, filename)
            image.save(image_path)

            data.append({
                "label": label,
                "confidence": confidence,
                "beschreibung": beschreibung,
                "fundort": fundort,
                "funddatum": str(funddatum),
                "image": image_path
            })

            save_data(data)
            st.success("✅ Fund erfolgreich gespeichert!")

# ==================================================
# TAB 2: Fund suchen
# ==================================================
with tab2:
    st.header("Verlorenen Gegenstand suchen")

    suchwort = st.selectbox(
        "Was suchst du?",
        ["Flasche", "Stift", "Brotdose"]
    )

    data = load_data()
    treffer = [d for d in data if d["label"] == suchwort]

    if treffer:
        st.subheader(f"Gefundene {suchwort}-Gegenstände:")
        for d in treffer:
            st.image(d["image"], width=250)
            st.write(f"**Beschreibung:** {d['beschreibung']}")
            st.write(f"**Fundort:** {d['fundort']}")
            st.write(f"**Datum:** {d['funddatum']}")
            st.write(f"**KI-Sicherheit:** {d['confidence']:.2%}")
            st.markdown("---")
    else:
        st.info("❌ Keine passenden Funde vorhanden.")
