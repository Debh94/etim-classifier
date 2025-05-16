import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF per leggere PDF
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

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
    righe_senza_sinonimi = df[df['Sinonimi'].str.strip() == '']
    if not righe_senza_sinonimi.empty:
        st.warning(f"🔍 Attenzione: {len(righe_senza_sinonimi)} classi non hanno sinonimi. Considera di arricchirle per migliorare la precisione.")

    df['combined_text'] = df.apply(lambda row: ' '.join([
        row['Description (EN)'], row['ETIM IT'], row['Translation (ETIM CH)'],
        row['Traduttore Google'], row['Traduzione_DEF'], row['Sinonimi']
    ]).lower(), axis=1)
    return df

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def classify_semantic_top_k(description, df, model, top_k=5):
    corpus = df['combined_text'].tolist()
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    input_embedding = model.encode(description, convert_to_tensor=True)
    similarities = cosine_similarity([input_embedding.cpu().numpy()], corpus_embeddings.cpu().numpy()).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results["Confidence"] = [round(similarities[i] * 100, 2) for i in top_indices]
    return results

# === Streamlit app ===
st.set_page_config(page_title="Classificatore ETIM", layout="centered")
st.title("🤖 Classificatore automatico ETIM da testo, PDF o URL")
st.markdown("Inserisci una **descrizione**, carica un **PDF** o incolla un **link** per identificare la classe ETIM.")

# Caricamento dati
df_etim = load_etim_data()
semantic_model = load_semantic_model()

# Input libero
user_input = st.text_area("📌 Oppure inserisci direttamente la descrizione del prodotto:", height=150)

# Caricamento PDF
pdf_file = st.file_uploader("📎 Carica una scheda tecnica in PDF (facoltativo):", type="pdf")
if pdf_file:
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text_pdf = " ".join(page.get_text() for page in doc)
            user_input = text_pdf
            if len(user_input) < 50:
                st.warning("⚠️ Attenzione: il testo estratto dal PDF è molto breve e potrebbe non essere sufficiente.")
    except Exception as e:
        st.error(f"Errore nella lettura del PDF: {e}")

# Inserimento URL
url_input = st.text_input("🔗 Oppure incolla un link a una scheda prodotto online:")
if url_input:
    try:
        response = requests.get(url_input, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'li'])
        text = ' '.join(p.get_text() for p in paragraphs if len(p.get_text()) > 30)
        user_input = text.strip()
        if len(user_input) < 50:
            st.warning("⚠️ Attenzione: il testo estratto dal link è molto breve e potrebbe non essere sufficiente.")
        st.text_area("🧾 Testo estratto dalla pagina:", value=user_input[:2000], height=150)
    except Exception as e:
        st.error(f"Errore nel caricamento della pagina: {e}")

# Classificazione finale
if st.button("Classifica"):
    if user_input.strip():
        st.markdown("🧠 Testo analizzato dall'AI:")
        st.code(user_input[:600])
        top_results = classify_semantic_top_k(user_input, df_etim, semantic_model, top_k=5)
        st.success("✅ Classi ETIM suggerite:")
        for _, row in top_results.iterrows():
            st.markdown(f"**{row['Code']}** – {row['ETIM IT']}  ")
            st.markdown(f"📈 Confidenza AI: {row['Confidence']}%")
            st.markdown("---")
    else:
        st.warning("Inserisci una descrizione, carica un PDF o un link.")
