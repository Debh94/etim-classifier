import os
# Disabilita il watcher di file che va in crash con torch.classes
os.environ["STREAMLIT_WATCH_FILES"] = "false"

import logging
logging.basicConfig(level=logging.DEBUG)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_etim_resources():
    # 1) Carica e pulisci il dataset
    df = pd.read_excel("Classi_9.xlsx", engine="openpyxl")
    cols = [
        "Code","Description (EN)","ETIM IT",
        "Translation (ETIM CH)","Traduttore Google",
        "Traduzione_DEF","Sinonimi"
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"❌ Mancano colonne nel file Excel: {missing}")
        st.stop()
    df = df[cols].fillna("")
    df["combined_text"] = df.apply(
        lambda r: " ".join([
            r["Description (EN)"],
            r["ETIM IT"],
            r["Translation (ETIM CH)"],
            r["Traduttore Google"],
            r["Traduzione_DEF"],
            r["Sinonimi"],
        ]).lower(),
        axis=1
    )

    # 2) Carica il modello e pre-calcola gli embedding del corpus
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    corpus_emb = model.encode(
        df["combined_text"].tolist(),
        convert_to_tensor=True
    )

    return df, model, corpus_emb

# qui carichiamo TUTTO con una sola call cachata
df_etim, model, corpus_emb = load_etim_resources()

# --- UI Streamlit ---
st.title("🤖 Classificatore automatico ETIM")
st.markdown("Inserisci la **descrizione** del prodotto per associare la classe ETIM corretta.")

user_input = st.text_area("📌 Descrizione del prodotto:", height=150)

if st.button("Classifica"):
    desc = user_input.strip_
