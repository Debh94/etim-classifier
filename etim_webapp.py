import logging
logging.basicConfig(level=logging.INFO)

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

st.set_page_config(page_title="ETIM AI Assistant", layout="centered")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_etim_data():
    df = pd.read_excel("Classi_9.xlsx", engine="openpyxl")
    df = df.fillna('')
    df['combined_text'] = df.apply(
        lambda row: ' '.join([
            row['Description (EN)'],
            row['ETIM IT'],
            row['Translation (ETIM CH)'],
            row['Traduttore Google'],
            row['Traduzione_DEF'],
            row['Sinonimi']
        ]).lower(), axis=1
    )
    return df

@st.cache_data
def embed_etim_classes(df):
    return model.encode(df['combined_text'].tolist(), convert_to_tensor=True)

model = load_model()
df_etim = load_etim_data()
corpus_embeddings = embed_etim_classes(df_etim)

tab1, tab2 = st.tabs(["📥 Classificatore", "🧠 Assistente AI"])

with tab1:
    st.title("📥 Classificatore ETIM")
    st.markdown("Inserisci una descrizione di prodotto per ricevere la **classe ETIM più adatta**.")

    user_input = st.text_area("✏️ Descrizione del prodotto:", height=150)
    if st.button("Classifica"):
        query = user_input.strip().lower()
        if not query:
            st.warning("⚠️ Inserisci una descrizione.")
        else:
            with st.spinner("🔍 Analisi semantica in corso..."):
                query_embedding = model.encode(query, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]

                results = []
                for hit in hits:
                    idx = hit['corpus_id']
                    score = round(float(hit['score']) * 100, 2)
                    row = df_etim.iloc[idx].copy()
                    row['Confidence'] = score
                    results.append(row)

                results_df = pd.DataFrame(results)

            if results_df.empty:
                st.error("❌ Nessun suggerimento trovato.")
            else:
                st.success("✅ Classi ETIM suggerite:")
                for _, r in results_df.iterrows():
                    st.markdown(f"""**{r['Code']}** – {r['ETIM IT']}  
🌍 *{r['Description (EN)']}*  
🇮🇹 Traduzioni: {r['Translation (ETIM CH)']}, {r['Traduttore Google']}, {r['Traduzione_DEF']}  
📊 Confidenza: {r['Confidence']}%""")
                    st.markdown("---")

with tab2:
    st.title("🧠 Assistente AI - Cos'è questo oggetto?")
    st.markdown("Scrivi una parola o frase per ottenere un'**interpretazione del significato** del prodotto.")

    ai_query = st.text_input("🔍 Cerca di che oggetto stiamo parlando:")

    if ai_query.strip():
        with st.spinner("🧠 Sto cercando di capire a cosa ti riferisci..."):
            query_embedding = model.encode(ai_query.strip().lower(), convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)[0]

            if not hits:
                st.warning("⚠️ Nessun risultato trovato.")
            else:
                idx = hits[0]['corpus_id']
                r = df_etim.iloc[idx]
                st.success("✅ Ecco cosa potresti intendere:")
                st.markdown(f"""**Oggetto interpretato:** {r['ETIM IT']}  
🌍 *Descrizione tecnica (EN)*: {r['Description (EN)']}  
🇮🇹 Traduzioni disponibili: {r['Translation (ETIM CH)']}, {r['Traduttore Google']}, {r['Traduzione_DEF']}""")
                st.markdown("👉 Copia questa interpretazione per usarla nel classificatore.")
