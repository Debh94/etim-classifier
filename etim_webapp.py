import logging
logging.basicConfig(level=logging.DEBUG)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Caricamento e preparazione dati ETIM ---
@st.cache_data
def load_etim_resources():
    try:
        df = pd.read_excel("Classi_9.xlsx", engine="openpyxl")
    except FileNotFoundError:
        st.error("❌ File 'Classi_9.xlsx' non trovato nella cartella dell'app.")
        st.stop()

    # Colonne richieste
    cols = [
        'Code', 'Description (EN)', 'ETIM IT',
        'Translation (ETIM CH)', 'Traduttore Google',
        'Traduzione_DEF', 'Sinonimi'
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"❌ Mancano colonne nel file Excel: {missing}")
        st.stop()

    # Compatta testo in una singola colonna
    df = df[cols].fillna('')
    df['combined_text'] = df.apply(
        lambda r: ' '.join([
            r['Description (EN)'], r['ETIM IT'],
            r['Translation (ETIM CH)'], r['Traduttore Google'],
            r['Traduzione_DEF'], r['Sinonimi']
        ]).lower(), axis=1
    )

    # Inizializza TF-IDF sul corpus
    vectorizer = TfidfVectorizer()
    corpus_tfidf = vectorizer.fit_transform(df['combined_text'])

    return df, vectorizer, corpus_tfidf

# Carica risorse (cached)
df_etim, vectorizer, corpus_tfidf = load_etim_resources()

# --- Interfaccia Streamlit ---
st.title("🤖 Classificatore automatico ETIM")
st.markdown("Inserisci la descrizione del prodotto per ottenere la classe ETIM corretta in tempo reale.")

user_input = st.text_area("📌 Descrizione del prodotto:", height=150)

if st.button("Classifica"):
    desc = user_input.strip().lower()
    if not desc:
        st.warning("⚠️ Inserisci una descrizione prima di procedere.")
    else:
        with st.spinner("🔍 Calcolo TF-IDF e similarità..."):
            # Trasformazione del testo utente
            desc_tfidf = vectorizer.transform([desc])
            sims = cosine_similarity(desc_tfidf, corpus_tfidf).flatten()
            top_idx = sims.argsort()[-5:][::-1]

            results = df_etim.iloc[top_idx].copy()
            results['Confidence'] = [round(float(s)*100,2) for s in sims[top_idx]]

        if results.empty:
            st.error("❌ Nessun suggerimento trovato. Controlla il file ETIM.")
        else:
            st.success("✅ Classi ETIM suggerite:")
            for _, r in results.iterrows():
                st.markdown(f"**{r['Code']}** – {r['ETIM IT']}  (Confidenza: {r['Confidence']}%)")
                st.markdown(f"🔤 Descrizione EN: {r['Description (EN)']}")
                st.markdown("---")

# --- Fine dell'app ---
