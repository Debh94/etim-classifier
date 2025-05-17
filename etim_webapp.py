import logging
logging.basicConfig(level=logging.DEBUG)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pkg_resources

# Mostra le librerie disponibili
installed = [p.key for p in pkg_resources.working_set]
st.write("✅ Librerie disponibili:", installed)

# === ETIM setup ===
@st.cache_data
def load_etim_data():
    """Carica dati ETIM da Excel e verifica colonne richieste."""
    try:
        df = pd.read_excel("Classi_9.xlsx", engine='openpyxl')
    except FileNotFoundError:
        st.error("❌ File 'Classi_9.xlsx' non trovato. Controlla la cartella.")
        st.stop()

    cols = [
        'Code', 'Description (EN)', 'ETIM IT',
        'Translation (ETIM CH)', 'Traduttore Google',
        'Traduzione_DEF', 'Sinonimi'
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"❌ Mancano colonne: {missing}")
        st.stop()

    df = df[cols].fillna('')
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
    """Carica modello leggero per embedding."""
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Calcola embeddings senza decorator per parametri hashable
def compute_embeddings(df):
    """Precalcola embeddings del corpus ETIM."""
    model = load_model()
    texts = df['combined_text'].tolist()
    return model.encode(texts, convert_to_tensor=True)

@st.cache_data
def get_corpus_embeddings():
    """Ritorna DataFrame ETIM e embeddings precomputati."""
    df = load_etim_data()
    emb = compute_embeddings(df)
    return df, emb

# Funzione di classificazione
def classify_etim(desc, df, corpus_emb, model, top_k=5):
    inp_emb = model.encode(desc.lower(), convert_to_tensor=True)
    sims = cosine_similarity(
        [inp_emb.cpu().numpy()],
        corpus_emb.cpu().numpy()
    ).flatten()
    idx = sims.argsort()[-top_k:][::-1]
    res = df.iloc[idx].copy()
    res['Confidence'] = [round(s * 100, 2) for s in sims[idx]]
    return res

# === Streamlit UI ===
st.title("🤖 Classificatore automatico ETIM")
st.markdown("Inserisci la descrizione del prodotto per ottenere la classe ETIM.")

# Carica dati ed embeddings
 # Carica insieme DataFrame e embeddings (cached)
df_etim, corpus_emb = get_corpus_embeddings()
# Carica il modello per le nuove descrizioni
model = load_model()


# Input dell'utente
user_input = st.text_area("📌 Descrizione del prodotto:", height=150)

if st.button("Classifica"):
    desc = user_input.strip()
    if not desc:
        st.warning("⚠️ Inserisci una descrizione.")
    else:
        with st.spinner("🔍 Classificazione in corso..."):
            results = classify_etim(desc, df_etim, corpus_emb, model)
        if results.empty:
            st.error("❌ Nessun suggerimento. Controlla il foglio ETIM.")
        else:
            st.success("✅ Classi ETIM suggerite:")
            for _, r in results.iterrows():
                st.markdown(f"**{r['Code']}** – {r['ETIM IT']} (Conf.: {r['Confidence']}%)")
                st.markdown(f"🔤 EN: {r['Description (EN)']}")
                st.markdown("---")

# Fine dell'app
