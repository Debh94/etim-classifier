import logging
logging.basicConfig(level=logging.INFO)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_etim_data():
    try:
        df = pd.read_excel("Classi_9.xlsx", engine="openpyxl")
    except FileNotFoundError:
        st.error("❌ File 'Classi_9.xlsx' non trovato nella cartella dell'app.")
        st.stop()

    cols = [
        'Code', 'Description (EN)', 'ETIM IT',
        'Translation (ETIM CH)', 'Traduttore Google',
        'Traduzione_DEF', 'Sinonimi'
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"❌ Mancano colonne nel file Excel: {missing}")
        st.stop()

    df = df[cols].fillna('')
    df['combined_text'] = df.apply(
        lambda r: ' '.join([
            r['Description (EN)'], r['ETIM IT'],
            r['Translation (ETIM CH)'], r['Traduttore Google'],
            r['Traduzione_DEF'], r['Sinonimi']
        ]).lower(), axis=1
    )
    return df

# Caricamento risorse
model = load_model()
df_etim = load_etim_data()

# Calcola gli embedding per tutte le classi
@st.cache_data
def embed_etim_classes(df):
    return model.encode(df['combined_text'].tolist(), convert_to_tensor=True)

corpus_embeddings = embed_etim_classes(df_etim)

# Interfaccia utente
st.title("🤖 Classificatore ETIM con AI")
st.markdown("Inserisci una descrizione di prodotto per ricevere la **classe ETIM più adatta** con un sistema semantico intelligente.")

user_input = st.text_area("📌 Descrizione del prodotto:", height=150)

if st.button("Classifica"):
    query = user_input.strip().lower()
    if not query:
        st.warning("⚠️ Inserisci una descrizione prima di procedere.")
    else:
        with st.spinner("🔍 Analisi semantica in corso..."):
            query_embedding = model.encode(query, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]

            st.success("✅ Classi ETIM suggerite:")
            for hit in hits:
                idx = hit['corpus_id']
                score = round(float(hit['score']) * 100, 2)
                row = df_etim.iloc[idx]
                st.markdown(f"**{row['Code']}** – {row['ETIM IT']} (Confidenza: {score}%)")
                st.markdown(f"🔤 Descrizione EN: {row['Description (EN)']}")
                st.markdown("---")
