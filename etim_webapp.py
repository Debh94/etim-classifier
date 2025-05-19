import logging
logging.basicConfig(level=logging.INFO)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_etim_data():
    df = pd.read_excel("Classi_9.xlsx", engine="openpyxl")
    cols = [
        'Code', 'Description (EN)', 'ETIM IT',
        'Translation (ETIM CH)', 'Traduttore Google',
        'Traduzione_DEF', 'Sinonimi'
    ]
    df = df[cols].fillna('')
    df['combined_text'] = df.apply(
        lambda r: ' '.join([
            r['Description (EN)'], r['ETIM IT'],
            r['Translation (ETIM CH)'], r['Traduttore Google'],
            r['Traduzione_DEF'], r['Sinonimi']
        ]).lower(), axis=1
    )
    return df

# Inizializzazione session_state per i feedback
if 'feedback' not in st.session_state:
    st.session_state.feedback = []

model = load_model()
df_etim = load_etim_data()

@st.cache_data
def embed_etim_classes(df):
    return model.encode(df['combined_text'].tolist(), convert_to_tensor=True)

corpus_embeddings = embed_etim_classes(df_etim)

st.title("🤖 Classificatore ETIM con AI")
st.markdown("Inserisci una descrizione di prodotto per ricevere la **classe ETIM più adatta** con un sistema semantico intelligente.")

user_input = st.text_area("📌 Descrizione del prodotto:", height=150)
classify = st.button("Classifica")

if classify and user_input.strip():
    query = user_input.strip().lower()
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

        results = pd.DataFrame(results)

    if results.empty:
        st.error("❌ Nessun suggerimento trovato.")
    else:
        st.success("✅ Classi ETIM suggerite:")
        for _, r in results.iterrows():
            st.markdown(f"**{r['Code']}** – {r['ETIM IT']} (Confidenza: {r['Confidence']}%)")
            st.markdown(f"🌍 Descrizione originale: {r['Description (EN)']}")
            st.markdown(f"🇮🇹 Traduzioni: {r['Translation (ETIM CH)']}, {r['Traduttore Google']}, {r['Traduzione_DEF']}")
            st.markdown("---")

        st.subheader("📣 Seleziona la classe corretta tra quelle suggerite")

        class_options = [
            f"{r['Code']} – {r['ETIM IT']} (Confidenza: {r['Confidence']}%)"
            for _, r in results.iterrows()
        ]
        selected = st.radio("🟢 Quale classe è corretta?", class_options, key="selected_class")
        commento = st.text_area("✏️ Commenti aggiuntivi (opzionale):", key="comment_input")
        invia_feedback = st.button("Invia feedback")

        if invia_feedback:
            idx = class_options.index(st.session_state.selected_class)
            r = results.iloc[idx]

            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "descrizione_utente": user_input,
                "classe_selezionata": r['Code'],
                "etim_it": r['ETIM IT'],
                "confidenza": r['Confidence'],
                "commento": st.session_state.comment_input,
                "classi_suggerite": "; ".join([c.split(" (")[0] for c in class_options])
            }

            st.session_state.feedback.append(feedback_data)
            st.success("✅ Feedback inviato correttamente!")

if st.session_state.feedback:
    st.markdown("### 📄 Feedback della sessione")
    st.dataframe(pd.DataFrame(st.session_state.feedback).tail(5))
