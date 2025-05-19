import logging
logging.basicConfig(level=logging.INFO)

import streamlit as st
st.set_page_config(page_title="Classificatore ETIM", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === GOOGLE SHEET SETUP ===
def save_feedback_to_google_sheet(feedback_row):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("google_credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("1PUJJKuMbbI4oyBtOTO6spDmstNeyLQXPMOAtdWK839U").sheet1

    sheet.append_row([
        feedback_row["timestamp"],
        feedback_row["descrizione_utente"],
        feedback_row["classe_selezionata"],
        feedback_row["etim_it"],
        feedback_row["confidenza"],
        feedback_row["commento"],
        feedback_row["classi_suggerite"]
    ])

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

            save_feedback_to_google_sheet(feedback_data)
            st.success("✅ Feedback inviato e salvato su Google Sheets!")
