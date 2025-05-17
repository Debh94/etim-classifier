import os
# Disabilita il watcher di file per evitare crash con torch
os.environ["STREAMLIT_WATCH_FILES"] = "false"

import logging
logging.basicConfig(level=logging.DEBUG)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Caricamento risorse ottimizzato ---
# Pre-requisito: eseguire una volta lo script `precompute_embeddings.py` per generare
# `etim_df.pkl` e `etim_embeddings.npy` (vedi commento in fondo).
@st.cache_data
def load_resources_offline():
    try:
        df = pd.read_pickle("etim_df.pkl")
        corpus_emb = np.load("etim_embeddings.npy")
    except FileNotFoundError as e:
        st.error(f"❌ File mancante: {e.filename}. Esegui `precompute_embeddings.py` prima.")
        st.stop()
    # Modello leggero per inferenza rapida
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, model, corpus_emb

# Carica una sola volta (cached)
df_etim, model, corpus_emb = load_resources_offline()

# --- Funzione di classificazione ---
def classify_etim(description: str, df: pd.DataFrame, corpus_emb: np.ndarray,
                   model: SentenceTransformer, top_k: int = 5) -> pd.DataFrame:
    desc = description.lower().strip()
    inp_emb = model.encode(desc, convert_to_tensor=False)
    sims = cosine_similarity([inp_emb], corpus_emb).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    res = df.iloc[top_idx].copy()
    res['Confidence'] = [round(float(s)*100,2) for s in sims[top_idx]]
    return res

# --- Interfaccia Streamlit ---
st.title("🤖 Classificatore ETIM (Offline Embeddings)")
st.info("⚙️ Caricamento dati, modello e embeddings (solo la prima volta può richiedere qualche secondo)...")

user_input = st.text_area("📌 Descrizione del prodotto:", height=150)
if st.button("Classifica"):
    if not user_input.strip():
        st.warning("⚠️ Inserisci una descrizione prima di procedere.")
    else:
        with st.spinner("🔍 Classificazione in corso..."):
            results = classify_etim(user_input, df_etim, corpus_emb, model)
        if results.empty:
            st.error("❌ Nessun suggerimento trovato. Controlla i file offline.")
        else:
            st.success("✅ Classi ETIM suggerite:")
            for _, r in results.iterrows():
                st.markdown(f"**{r['Code']}** – {r['ETIM IT']} (Conf.: {r['Confidence']}%)")
                st.markdown(f"🔤 EN: {r['Description (EN)']}")
                st.markdown("---")

# --- Script di preprocessing (esegui una sola volta) ---
#
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
# df = pd.read_excel("Classi_9.xlsx", engine="openpyxl")
# df['combined_text'] = df.apply(lambda r: " ".join([
#     r['Description (EN)'], r['ETIM IT'], r['Translation (ETIM CH)'],
#     r['Traduttore Google'], r['Traduzione_DEF'], r['Sinonimi']
# ]).lower(), axis=1)
#
# model = SentenceTransformer("all-MiniLM-L6-v2")
# emb = model.encode(df['combined_text'].tolist(), convert_to_tensor=False)
# np.save("etim_embeddings.npy", emb)
# df.to_pickle("etim_df.pkl")
# print("✅ Precompute completato: etim_embeddings.npy & etim_df.pkl pronti.")
