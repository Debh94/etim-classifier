import logging
logging.basicConfig(level=logging.INFO)

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import openai

st.set_page_config(page_title="Classificatore ETIM con AI", layout="centered")

# === MODELLO SEMANTICO ===
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

# === CARICAMENTO RISORSE ===
model = load_model()
df_etim = load_etim_data()
corpus_embeddings = embed_etim_classes(df_etim)

# === TABS INTERFACCIA ===
tab1, tab2, tab3 = st.tabs(["📥 Classificatore", "🔍 Assistente semantico", "🧠 ChatGPT"])

# === TAB 1: CLASSIFICATORE ===
with tab1:
    st.title("📥 Classificatore ETIM")
    st.markdown("Inserisci una descrizione di prodotto per ricevere la **classe ETIM più adatta**.")

    user_input = st.text_area("✏️ Descrizione del prodotto:", height=150)
    if st.button("Classifica", key="classifica_btn"):
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

# === TAB 2: ASSISTENTE SEMANTICO ===
with tab2:
    st.title("🔍 Assistente semantico")
    st.markdown("Scrivi una parola o frase per ottenere la **classe ETIM semanticamente più vicina**.")

    ai_query = st.text_input("🔎 Cerca un termine tecnico o descrizione generica:", key="query_semantic")
    if ai_query.strip():
        with st.spinner("🔍 Analisi in corso..."):
            query_embedding = model.encode(ai_query.strip().lower(), convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]

            if not hits:
                st.warning("⚠️ Nessun risultato trovato.")
            else:
                st.subheader("📘 Risultati semantici")
                for hit in hits:
                    idx = hit['corpus_id']
                    r = df_etim.iloc[idx]
                    st.markdown(f"""**{r['Code']}** – {r['ETIM IT']}  
🌍 *{r['Description (EN)')}*  
🇮🇹 Traduzioni: {r['Translation (ETIM CH)']}, {r['Traduttore Google']}, {r['Traduzione_DEF']}""")
                    st.markdown("---")

# === TAB 3: CHATGPT INTELLIGENTE ===
with tab3:
    st.title("🧠 Assistente AI - ChatGPT")
    st.markdown("Scrivi una descrizione o un nome prodotto: l'assistente GPT ti spiega cos'è e come cercarlo nel classificatore ETIM.")

    if "OPENAI_API_KEY" not in st.secrets:
        st.error("❌ API key OpenAI mancante. Inseriscila nei Secrets con chiave 'OPENAI_API_KEY'.")
    else:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        gpt_query = st.text_input("💬 Descrizione da interpretare:", key="query_gpt")
        if gpt_query.strip():
            with st.spinner("🤖 Sto interpretando l'oggetto..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "Sei un assistente tecnico esperto di classificazione ETIM."},
                            {"role": "user", "content": f"""
L'utente vuole capire meglio questo oggetto: "{gpt_query.strip()}".

1. Dai una breve definizione tecnica.
2. Spiega come si usa.
3. Suggerisci dove cercarlo nel classificatore ETIM (es. nome famiglia o funzione).

Usa un linguaggio chiaro ma tecnico.
"""}
                        ],
                        temperature=0.4
                    )

                    risposta = response.choices[0].message.content
                    st.success("✅ Ecco la spiegazione intelligente:")
                    st.markdown(risposta)

                except Exception as e:
                    st.error(f"Errore nella risposta GPT: {e}")
