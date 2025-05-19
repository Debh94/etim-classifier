import logging
logging.basicConfig(level=logging.INFO)

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import wikipedia
from datetime import datetime

st.set_page_config(page_title="GianPieTro", layout="centered")

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

tab1, tab2 = st.tabs(["GianPieTro", "Assistente Wikipedia"])

with tab1:
    st.title("🤖 GianPieTro - Classificatore ETIM")
    st.markdown("Benvenuto! Sono GianPieTro, il tuo aiutante per trovare la giusta classe ETIM.")
    st.markdown("Inserisci una descrizione di prodotto per ricevere la classe ETIM più adatta.")

    user_input = st.text_area("Descrizione del prodotto:", height=150)
    if st.button("Classifica", key="classifica_btn"):
        query = user_input.strip().lower()
        if not query:
            st.warning("Inserisci una descrizione.")
        else:
            with st.spinner("Analisi semantica in corso..."):
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
                st.error("Nessun suggerimento trovato.")
            else:
                st.success("Classi ETIM suggerite:")
                for _, r in results_df.iterrows():
                    st.markdown(f"""**{r["Code"]}** - {r["ETIM IT"]}  
🌍 Descrizione EN: {r["Description (EN)"]}  
🇮🇹 Traduzioni: {r["Translation (ETIM CH)"]}, {r["Traduttore Google"]}, {r["Traduzione_DEF"]}  
📊 Confidenza: {r["Confidence"]}%""")

with tab2:
    st.title("GianPieTro - il tuo assistente gratuito")
    st.markdown("Scrivi il nome di un oggetto per ricevere una definizione automatica da Wikipedia.")

    with st.form("wiki_form"):
        term = st.text_input("Oggetto da cercare:", key="term_wiki")
        cerca = st.form_submit_button("Cerca definizione")

    if cerca and term.strip():
        try:
            wikipedia.set_lang("it")
            summary = wikipedia.summary(term.strip(), sentences=3, auto_suggest=True, redirect=True)
            st.success("Ecco cosa ho trovato:")
            st.markdown(summary)
        except wikipedia.exceptions.DisambiguationError as e:
            st.warning("Il termine è ambiguo. Prova a essere più preciso. Esempi:")
            st.markdown(", ".join(e.options[:5]))
        except wikipedia.exceptions.PageError:
            st.error("Nessuna definizione trovata. Prova con un altro termine.")
        except Exception as ex:
            st.error(f"Errore durante la ricerca: {ex}")
