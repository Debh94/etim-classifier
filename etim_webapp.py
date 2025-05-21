import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import wikipedia
import torch

# Impostazioni pagina - deve essere il primo comando
st.set_page_config(page_title="GianPieTro", layout="centered")

# Applica lo stile grafico migliorato
st.markdown("""
    <style>
        .stTextInput > div > div > input {
            font-size: 14px;
        }
        .stTextArea > div > div > textarea {
            font-size: 14px;
        }
        .stButton > button {
            font-size: 14px;
            padding: 0.4em 1em;
        }
        h1, h2, h3 {
            font-size: 22px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Importa i dizionari aggiornati
from synonym_to_class import synonym_to_class
from fallback_value_to_class import fallback_mapping

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_etim_data():
    df = pd.read_excel("Classi_9.xlsx", engine="openpyxl")
    df = df.fillna("")
    df["combined_text"] = df.apply(
        lambda row: " ".join([
            row["Description (EN)"],
            row["ETIM IT"],
            row["Translation (ETIM CH)"],
            row["Traduttore Google"],
            row["Traduzione_DEF"],
            row["Sinonimi"]
        ]).lower(),
        axis=1
    )
    return df

@st.cache_data
def embed_etim_classes(df):
    return model.encode(df["combined_text"].tolist(), convert_to_tensor=True)

def normalize(txt):
    return txt.strip().lower()

model = load_model()
df_etim = load_etim_data()
corpus_embeddings = embed_etim_classes(df_etim)

tab1, tab2 = st.tabs(["GianPieTro", "Assistente Wikipedia"])

with tab1:
    st.title("🤖 GianPieTro - Classificatore ETIM")
    user_input = st.text_area("📌 Descrizione del prodotto:", height=150)

    if st.button("Classifica"):
        query = normalize(user_input)

        if query in synonym_to_class:
            st.success("✅ Trovato nei sinonimi:")
            for cl in sorted(set(synonym_to_class[query])):
                st.markdown(f"- Classe ETIM: **{cl}**")

        elif query in fallback_mapping:
            st.success("✅ Trovato nel fallback ETIM (value):")
            for cl in sorted(set(fallback_mapping[query])):
                st.markdown(f"- Classe ETIM: **{cl}**")

        else:
            st.info("🧠 Nessun match diretto. Cerco con AI...")
            query_embedding = model.encode(query, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]

            results = []
            for hit in hits:
                idx = hit["corpus_id"]
                score = round(float(hit["score"]) * 100, 2)
                row = df_etim.iloc[idx].copy()
                row["Confidence"] = score
                results.append(row)

            results_df = pd.DataFrame(results)

            if results_df.empty:
                st.error("❌ Nessun risultato trovato.")
            else:
                st.success("✅ Risultati AI:")
                for _, r in results_df.iterrows():
                    st.markdown(f"""**{r['Code']}** – {r['ETIM IT']}
📘 EN: {r['Description (EN)']}
🇮🇹 Traduzioni: {r['Translation (ETIM CH)']}, {r['Traduttore Google']}, {r['Traduzione_DEF']}
📊 Confidenza: {r['Confidence']}%
---""")

with tab2:
    st.title("📚 Assistente Wikipedia")
    with st.form("wiki_form"):
        term = st.text_input("Cerca un oggetto:", key="term_wiki")
        btn = st.form_submit_button("Cerca definizione")

    if btn and term.strip():
        try:
            wikipedia.set_lang("it")
            summary = wikipedia.summary(term.strip(), sentences=3)
            st.success("✅ Definizione trovata:")
            st.markdown(summary)
        except wikipedia.exceptions.DisambiguationError as e:
            st.warning("⚠️ Termine ambiguo. Esempi: " + ", ".join(e.options[:5]))
        except wikipedia.exceptions.PageError:
            st.error("❌ Nessuna definizione trovata.")
        except Exception as e:
            st.error(f"Errore: {e}")
