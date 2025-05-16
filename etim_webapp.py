import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF per leggere PDF
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# === ETIM setup ===
@st.cache_data
def load_etim_data():
    df = pd.read_excel("Classi_9.xlsx")

    required_cols = ['Code', 'Description (EN)', 'ETIM IT',
                     'Translation (ETIM CH)', 'Traduttore Google',
                     'Traduzione_DEF', 'Sinonimi']

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Mancano queste colonne nel file Excel: {missing}")
        st.stop()

    df = df[required_cols].fillna('')
    df['combined_text'] = df.apply(lambda row: ' '.join([
        row['Description (EN)'], row['ETIM IT'], row['Translation (ETIM CH)'],
        row['Traduttore Google'], row['Traduzione_DEF'], row['Sinonimi']
    ]).lower(), axis=1)
    return df

@st.cache_resource
def setup_classifier(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, tfidf_matrix

def classify_description(description, df, vectorizer, tfidf_matrix):
    input_vec = vectorizer.transform([description.lower()])
    similarity = cosine_similarity(input_vec, tfidf_matrix).flatten()
    idx = similarity.argmax()
    result = df.iloc[idx]
    return result, round(similarity[idx] * 100, 2)

# === BLIP model setup ===
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image_file, processor, model):
    image = Image.open(image_file).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# === Streamlit app ===
st.set_page_config(page_title="Classificatore ETIM", layout="centered")
st.title("🤖 Classificatore automatico ETIM da testo, PDF, URL o immagine")
st.markdown("Inserisci una **descrizione**, carica un **PDF**, incolla un **link** o carica una **foto** per identificare la classe ETIM.")

# Caricamento dati
df_etim = load_etim_data()
vectorizer, tfidf_matrix = setup_classifier(df_etim)

# Input libero
user_input = st.text_area("📌 Oppure inserisci direttamente la descrizione del prodotto:", height=150)

# Caricamento PDF
pdf_file = st.file_uploader("📎 Carica una scheda tecnica in PDF (facoltativo):", type="pdf")
if pdf_file:
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text_pdf = " ".join(page.get_text() for page in doc)
            user_input = text_pdf
    except Exception as e:
        st.error(f"Errore nella lettura del PDF: {e}")

# Inserimento URL
url_input = st.text_input("🔗 Oppure incolla un link a una scheda prodotto online:")
if url_input:
    try:
        response = requests.get(url_input, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        user_input = soup.get_text(separator=' ', strip=True)
    except Exception as e:
        st.error(f"Errore nel caricamento della pagina: {e}")

# Caricamento immagine + descrizione automatica (BLIP)
image_file = st.file_uploader("📷 Oppure carica una foto dell'articolo (jpg, png):", type=["jpg", "jpeg", "png"])
if image_file:
    try:
        st.image(image_file, caption="Immagine caricata", use_container_width=True)
        with st.spinner("Sto analizzando l'immagine..."):
            processor, model = load_blip_model()
            label = generate_image_caption(image_file, processor, model)
            user_input = label + ". " + user_input
            st.info(f"🧠 Descrizione generata automaticamente: **{label}**")
    except Exception as e:
        st.error(f"Errore durante l'elaborazione dell'immagine: {e}")

# Classificazione finale
if st.button("Classifica"):
    if user_input.strip():
        result, score = classify_description(user_input, df_etim, vectorizer, tfidf_matrix)
        st.success(f"✅ Classe ETIM suggerita: **{result['Code']}**")
        st.markdown(f"**Nome (EN):** {result['Description (EN)']}")
        st.markdown(f"**Nome (IT):** {result['ETIM IT']}")
        st.markdown(f"**Sinonimi/Traduzioni:**")
        st.markdown(f"- ETIM CH: {result['Translation (ETIM CH)']}")
        st.markdown(f"- Google Translate: {result['Traduttore Google']}")
        st.markdown(f"- Traduzione DEF: {result['Traduzione_DEF']}")
        st.markdown(f"**Confidenza AI:** {score}%")
    else:
        st.warning("Inserisci una descrizione, carica un PDF, un link o un'immagine.")
