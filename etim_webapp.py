import os
# Disabilita il file-watcher che causa errori su torch.classes
os.environ["STREAMLIT_WATCH_FILES"] = "false"

import logging
logging.basicConfig(level=logging.DEBUG)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Unico punto cached che non prende argomenti non-hashable
@st.cache_resource
def load_etim_resources():
    # 1) Carica e pulisci il file Excel
    try:
        df = pd.read_excel("Classi_9.xlsx", engine="openpyxl")
    except FileNotFoundError:
        st.error("❌ File 'Classi_9.xlsx' non trovato. Aggiungilo nella cartella dell'app.")
        st.stop()

    cols = [
        "Code","Description (EN)","ETIM IT",
        "Translation (ETIM CH)","Traduttore Google",
        "Traduzione_DEF","Sinonimi"
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"❌ Mancano queste colonne nel file Excel: {missing}")
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

    # 2) Instanzia il modello (leggero) e precalcola embeddings
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    corpus_emb = model.encode(
        df["combined_text"].tolist(),
        convert_to_tensor=True
    )

    return df, model, corpus_emb

# Carica TUTTO in un colpo solo
df_etim, model, corpus_emb = load_etim_resources()

# UI
st.title("🤖 Classificatore automatico ETIM")
st.markdown("Inserisci la **descrizione** del prodotto per ottenere la classe ETIM corretta.")

user_input = st.text_area("📌 Descrizione del prodotto:", height=150)

if st.button("Classifica"):
    desc = user_input.strip()
    if not desc:
        st.warning("⚠️ Inserisci una descrizione prima di procedere.")
    else:
        with st.spinner("🔍 Classificazione in corso..."):
            inp_emb = model.encode(desc.lower(), convert_to_tensor=True)
            sims = cosine_similarity(
                [inp_emb.cpu().numpy()],
                corpus_emb.cpu().numpy()
            ).flatten()
            top_idx = sims.argsort()[-5:][::-1]

            results = df_etim.iloc[top_idx].copy()
            results["Confidence"] = [round(float(s)*100, 2) for s in sims[top_idx]]

        if results.empty:
            st.error("❌ Nessun suggerimento trovato. Verifica il file ETIM.")
        else:
            st.success("✅ Classi ETIM suggerite:")
            for _, r in results.iterrows():
                st.markdown(f"**{r['Code']}** – {r['ETIM IT']} (Conf.: {r['Confidence']}%)")
                st.markdown(f"🔤 EN: {r['Description (EN)']}")
                st.markdown("---")
