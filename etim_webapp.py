import logging
logging.basicConfig(level=logging.DEBUG)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pkg_resources

# Mostra le librerie disponibili\installed = [p.key for p in pkg_resources.working_set]
st.write("✅ Librerie disponibili:", installed)

# === ETIM setup ===
@st.cache_data
def load_etim_data():
    try:
        df = pd.read_excel("Classi_9.xlsx", engine='openpyxl')
    except FileNotFoundError:
        st.error("❌ File 'Classi_9.xlsx' non trovato. Controlla che sia presente nella cartella.")
        st.stop()

    required_cols = [
        'Code', 'Description (EN)', 'ETIM IT',
        'Translation (ETIM CH)', 'Traduttore Google',
        'Traduzione_DEF', 'Sinonimi'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Mancano queste colonne nel file Excel: {missing}")
        st.stop()

    df = df[required_cols].fillna('')
    df['combined_text'] = df.apply(
        lambda r: ' '.join([
            r['Description (EN)'], r['ETIM IT'], r['Translation (ETIM CH)'],
            r['Traduttore Google'], r['Traduzione_DEF'], r['Sinonimi']
        ]).lower(),
        axis=1
    )
    return df

@st.cache_resource
def load_model():
    # Modello più leggero per prestazioni migliori
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Calcola embeddings senza caching per evitare errori di parametri non hashable
def compute_embeddings(df, model):
    corpus = df['combined_text'].tolist()
    return model.encode(corpus, convert_to_tensor=True)

# Funzione di classificazione semplificata
def classify_etim(description, df, model, corpus_emb, top_k=5):
    input_emb = model.encode(description.lower(), convert_to_tensor=True)
    sims = cosine_similarity(
        [input_emb.cpu().numpy()],
        corpus_emb.cpu().numpy()
    ).flatten()
    idx = sims.argsort()[-top_k:][::-1]
    results = df.iloc[idx].copy()
    results['Confidence'] = [round(s * 100, 2) for s in sims[idx]]
    return results

# === Streamlit app ===
st.title("🤖 Classificatore automatico ETIM")
st.markdown(
    "Inserisci la descrizione del prodotto e ottieni la classe ETIM corretta."
)

# Caricamento dati e modelli
df_etim = load_etim_data()
model = load_model()
corpus_emb = compute_embeddings(df_etim, model)

# Input descrizione semplice
user_input = st.text_area(
    "📌 Descrizione del prodotto:", height=150
)

if st.button("Classifica"):
    desc = user_input.strip()
    if not desc:
        st.warning("⚠️ Inserisci una descrizione prima di classificare.")
    else:
        with st.spinner("🔍 Classificazione in corso..."):
            top_results = classify_etim(desc, df_etim, model, corpus_emb)
        if top_results.empty:
            st.error("❌ Nessuna classe suggerita. Controlla il foglio ETIM.")
        else:
            st.success("✅ Classi ETIM suggerite:")
            for _, row in top_results.iterrows():
                st.markdown(f"**{row['Code']}** – {row['ETIM IT']} (Confidenza: {row['Confidence']}%)")
                st.markdown(f"🔤 Descrizione EN: {row['Description (EN)']}  ")
                st.markdown("---")

# Fine dell'app
