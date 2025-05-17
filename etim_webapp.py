import logging
logging.basicConfig(level=logging.DEBUG)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
import pkg_resources
from sklearn.metrics.pairwise import cosine_similarity

# Mostra le librerie disponibili per debug
installed = [p.key for p in pkg_resources.working_set]
st.write("✅ Librerie disponibili:", installed)

# === Funzioni di caricamento ===
@st.cache_data
def load_etim_data():
    """Carica il file Excel e prepara il DataFrame."""
    try:
        df = pd.read_excel("Classi_9.xlsx", engine='openpyxl')
    except FileNotFoundError:
        st.error("❌ File 'Classi_9.xlsx' non trovato nella cartella. Aggiungilo prima di eseguire l'app.")
        st.stop()

    cols = ['Code', 'Description (EN)', 'ETIM IT',
            'Translation (ETIM CH)', 'Traduttore Google',
            'Traduzione_DEF', 'Sinonimi']
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

@st.cache_resource
def load_model():
    """Carica un modello leggero per embedding."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Nessun decorator per questa funzione per evitare problemi di hash
def compute_embeddings(model, df):
    """Precalcola embeddings del corpus ETIM usando il modello fornito."""
    texts = df['combined_text'].tolist()
    return model.encode(texts, convert_to_tensor=True)

# === Caricamento iniziale dati e modelli ===
# Carica dataset ETIM e modello
 df_etim = load_etim_data()
model = load_model()
# Precalcola embeddings del corpus
corpus_emb = compute_embeddings(model, df_etim)

# === Funzione di classificazione ===
def classify_etim(description, df, corpus_emb, model, top_k=5):
    """Ritorna le migliori top_k classi ETIM con confidenza percentuale."""
    inp_emb = model.encode(description.lower(), convert_to_tensor=True)
    sims = cosine_similarity([inp_emb.cpu().numpy()], corpus_emb.cpu().numpy()).flatten()
    idx = sims.argsort()[-top_k:][::-1]
    res = df.iloc[idx].copy()
    res['Confidence'] = [round(float(s) * 100, 2) for s in sims[idx]]
    return res

# === Interfaccia Streamlit ===
st.title("🤖 Classificatore automatico ETIM")
st.markdown("Inserisci la descrizione del prodotto per ottenere la classe ETIM corretta.")

# Input descrizione prodotto
user_input = st.text_area("📌 Descrizione del prodotto:", height=150)

if st.button("Classifica"):
    desc = user_input.strip()
    if not desc:
        st.warning("⚠️ Inserisci una descrizione prima di continuare.")
    else:
        with st.spinner("🔍 Classificazione in corso..."):
            results = classify_etim(desc, df_etim, corpus_emb, model)
        if results.empty:
            st.error("❌ Nessun suggerimento trovato. Verifica il file ETIM e i dati.")
        else:
            st.success("✅ Classi ETIM suggerite:")
            for _, r in results.iterrows():
                st.markdown(f"**{r['Code']}** – {r['ETIM IT']} (Confidenza: {r['Confidence']}%)")
                st.markdown(f"🔤 Descrizione EN: {r['Description (EN)']}")
                st.markdown("---")

# Fine dell'app
