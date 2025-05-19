import logging
logging.basicConfig(level=logging.INFO)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import os

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

@st.cache_data
def load_feedback():
    if os.path.exists("feedback.csv"):
        df_fb = pd.read_csv("feedback.csv")
        df_fb['combined_text'] = df_fb['descrizione_utente'].str.lower()
        df_fb['Code'] = df_fb['classe_selezionata']
        df_fb['ETIM IT'] = df_fb['etim_it']
        return df_fb[['combined_text', 'Code', 'ETIM IT']]
    return pd.DataFrame(columns=['combined_text', 'Code', 'ETIM IT'])

# Caricamento risorse
model = load_model()
df_etim = load_etim_data()
df_feedback = load_feedback()
df_combined = pd.concat([df_etim[['combined_text', 'Code', 'ETIM IT']], df_feedback], ignore_index=True)

@st.cache_data
def embed_etim_classes(df):
    return model.encode(df['combined_text'].tolist(), convert_to_tensor=True)

corpus_embeddings = embed_etim_classes(df_combined)

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

            results = []
            for hit in hits:
                idx = hit['corpus_id']
                score = round(float(hit['score']) * 100, 2)
                row = df_combined.iloc[idx].copy()
                row['Confidence'] = score
                results.append(row)

            results = pd.DataFrame(results)

        if results.empty:
            st.error("❌ Nessun suggerimento trovato.")
        else:
            st.success("✅ Classi ETIM suggerite:")
            for _, r in results.iterrows():
                st.markdown(f"**{r['Code']}** – {r['ETIM IT']} (Confidenza: {r['Confidence']}%)")
                st.markdown("---")

            st.subheader("📣 Seleziona la classe corretta tra quelle suggerite")

            class_options = [
                f"{r['Code']} – {r['ETIM IT']} (Confidenza: {r['Confidence']}%)"
                for _, r in results.iterrows()
            ]
            selected = st.radio("🟢 Quale classe è corretta?", class_options)

            commento = st.text_area("✏️ Commenti aggiuntivi (opzionale):")

            if st.button("Invia feedback"):
                idx = class_options.index(selected)
                r = results.iloc[idx]

                feedback_data = {
                    "timestamp": datetime.now().isoformat(),
                    "descrizione_utente": user_input,
                    "classe_selezionata": r['Code'],
                    "etim_it": r['ETIM IT'],
                    "confidenza": r['Confidence'],
                    "commento": commento,
                    "classi_suggerite": "; ".join([c.split(" (")[0] for c in class_options])
                }

                feedback_df = pd.DataFrame([feedback_data])
                feedback_path = "feedback.csv"
                if os.path.exists(feedback_path):
                    feedback_df.to_csv(feedback_path, mode='a', header=False, index=False)
                else:
                    feedback_df.to_csv(feedback_path, index=False)

                st.success("✅ Feedback inviato correttamente e salvato nel file!")
