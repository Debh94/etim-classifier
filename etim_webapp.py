import os
os.environ["STREAMLIT_WATCH_FILES"] = "false"  # disabilita il watcher su torch

import logging
logging.basicConfig(level=logging.DEBUG)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1) Carica e pulisci i dati ETIM
def load_etim_data():
    df = pd.read_excel("Classi_9.xlsx", engine="openpyxl")
    cols = ["Code","Description (EN)","ETIM IT",
            "Translation (ETIM CH)","Traduttore Google",
            "Traduzione_DEF","Sinonimi"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"❌ Mancano colonne: {missing}")
        st.stop()
    df = df[cols].fillna("")
    df["combined_text"] = df.apply(
        lambda r: " ".join([
            r["Description (EN)"], r["ETIM IT"],
            r["Translation (ETIM CH)"], r["Traduttore Google"],
            r["Traduzione_DEF"], r["Sinonimi"]
        ]).lower(),
        axis=1
    )
    return df

# 2) Carica il modello
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 3) Pre-calcola gli embeddings
def compute_corpus_embeddings(model, df):
    return model.encode(df["combined_text"].tolist(), convert_to_tensor=True)

# Caricamento in chiaro (no cache)
df_etim   = load_etim_data()
model     = load_model()
corpus_emb = compute_corpus_embeddings(model, df_etim)

# Interfaccia Streamlit
st.title("🤖 Classificatore automatico ETIM")
st.markdown("Inserisci la descrizione del prodotto per ottenere la classe ETIM corretta.")

user_input = st.text_area("📌 Descrizione del prodotto:", height=150)

if st.button("Classifica"):
    desc = user_input.strip()
    if not desc:
        st.warning("⚠️ Inserisci una descrizione.")
    else:
        with st.spinner("🔍 Classificazione in corso..."):
            inp_emb = model.encode(desc.lower(), convert_to_tensor=True)
            sims    = cosine_similarity([inp_emb.cpu().numpy()],
                                        corpus_emb.cpu().numpy()).flatten()
            top_idx = sims.argsort()[-5:][::-1]
            results = df_etim.iloc[top_idx].copy()
            results["Confidence"] = [round(float(s)*100,2) for s in sims[top_idx]]

        if results.empty:
            st.error("❌ Nessun suggerimento trovato.")
        else:
            st.success("✅ Classi ETIM suggerite:")
            for _, r in results.iterrows():
                st.markdown(f"**{r['Code']}** – {r['ETIM IT']} (Conf.: {r['Confidence']}%)")
                st.markdown(f"🔤 EN: {r['Description (EN)']}")
                st.markdown("---")
