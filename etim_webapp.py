import streamlit as st
import pandas as pd
import sqlite3
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import wikipedia
import torch

# Impostazioni pagina
st.set_page_config(page_title="GianPieTro", layout="centered")

# Applica stile compatto
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

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_database():
    conn = sqlite3.connect("etim_classifier.db")
    class_df = pd.read_sql_query("SELECT * FROM class", conn)
    synonym_df = pd.read_sql_query("SELECT * FROM synonym_map", conn)
    value_df = pd.read_sql_query("SELECT * FROM value", conn)
    cfv_map_df = pd.read_sql_query("SELECT * FROM class_feature_value_map", conn)
    cff_map_df = pd.read_sql_query("SELECT * FROM class_feature_map", conn)
    conn.close()
    return class_df, synonym_df, value_df, cfv_map_df, cff_map_df

@st.cache_data
def embed_etim_classes(df):
    return model.encode(df["ARTCLASSDESC"].str.lower().tolist(), convert_to_tensor=True)

def normalize(txt):
    return txt.strip().lower()

model = load_model()
df_etim, df_synonyms, df_values, df_cfv, df_cff = load_database()
corpus_embeddings = embed_etim_classes(df_etim)

tab1, tab2 = st.tabs(["GianPieTro", "Assistente Wikipedia"])

with tab1:
    st.title("🤖 GianPieTro - Classificatore ETIM")
    user_input = st.text_area("📌 Descrizione del prodotto:", height=150)

    if st.button("Classifica"):
        query = normalize(user_input)

        # Cerca corrispondenze dirette come VALUE
        matched_values = df_values[df_values['VALUEDESC'].str.lower() == query]

        # Se non troviamo una corrispondenza in italiano, proviamo a cercare nella colonna inglese
        if matched_values.empty and 'TRANSLATION' in df_values.columns:
            matched_values = df_values[df_values['TRANSLATION'].str.lower() == query]

        if not matched_values.empty:
            st.success("🎯 Trovato come VALUE ETIM:")
            value_id = matched_values.iloc[0]['VALUEID']
            italian_translation = matched_values.iloc[0].get('VALUEDESC', '')
            english_desc = matched_values.iloc[0].get('TRANSLATION', '')
            st.markdown(f"- **{value_id}** – {italian_translation} 🇮🇹 / {english_desc} 🇬🇧")

            # Trova tutte le classi collegate a questo VALUE
            cfv_matches = df_cfv[df_cfv['VALUEID'] == value_id]
            artclass_ids = df_cff[df_cff['ARTCLASSFEATURENR'].isin(cfv_matches['ARTCLASSFEATURENR'])]['ARTCLASSID'].unique()
            st.markdown("\n📚 Classi ETIM che usano questo valore:")
            for cl in artclass_ids:
                desc = df_etim[df_etim['ARTCLASSID'] == cl]['ARTCLASSDESC'].values[0]
                st.markdown(f"- **{cl}** – {desc}")
        else:
            # Fallback: ricerca nei sinonimi
            matched_classes = df_synonyms[df_synonyms['CLASSSYNONYM'].str.lower().str.contains(query)]['ARTCLASSID'].unique()

            if len(matched_classes) > 0:
                st.success("✅ Trovato nei sinonimi:")
                for cl in sorted(matched_classes):
                    desc = df_etim[df_etim['ARTCLASSID'] == cl]['ARTCLASSDESC'].values[0]
                    st.markdown(f"- **{cl}** – {desc}")
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
                        st.markdown(f"""**{r['ARTCLASSID']}** – {r['ARTCLASSDESC']}
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
