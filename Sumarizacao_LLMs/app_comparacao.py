import streamlit as st
import time
import pandas as pd
import os
import random

# Modelos e ROUGE
from models.bart_model import BartSummarizer
from models.gpt_model import GPTSummarizer
from models.llama_model import LlamaSummarizer
from models.deepseek_local_model import DeepseekSummarizer
from models.bart_finetuned_model import BartFineTunedSummarizer
from evaluate_rouge import evaluate_summary

# Para tradução
from transformers import pipeline

##############################################################################
# 1) Configurações de Layout e Título
##############################################################################
st.set_page_config(
    page_title="Comparador de Modelos de Sumarização",
    layout="wide"
)
st.title("Comparador de Modelos de Sumarização")

##############################################################################
# 2) Pipeline de Tradução (EN->PT)
##############################################################################
@st.cache_resource
def load_translator():
    return pipeline("translation_en_to_pt", model="Helsinki-NLP/opus-mt-tc-big-en-pt")

translator = load_translator()

def translate_to_pt(text):
    if not text.strip():
        return ""
    result = translator(text, max_length=512)
    return result[0]["translation_text"]

##############################################################################
# 3) Carregar Dataset CNN/DailyMail
##############################################################################
@st.cache_data
def load_cnn_dataset():
    return pd.read_csv("data/cnn_dailymail/test.csv")

df_cnn = load_cnn_dataset()

# Barra lateral
st.sidebar.title("Configurações")
option = st.sidebar.radio(
    "Selecione a Fonte do Artigo:",
    ["Usar artigo aleatório da base", "Colar manualmente"]
)

# Variáveis de artigo e resumo humano
article = ""
reference_summary = ""

if "has_random_article" not in st.session_state:
    st.session_state.has_random_article = False

##############################################################################
# 4) Botão para Gerar Artigo Aleatório
##############################################################################
if option == "Usar artigo aleatório da base":
    if st.sidebar.button("Gerar Artigo Aleatório"):
        row = df_cnn.sample(1).iloc[0]
        st.session_state.article_random = row["article"]
        st.session_state.ref_summary_random = row["highlights"]
        st.session_state.has_random_article = True

    if st.session_state.has_random_article:
        article = st.session_state.article_random
        reference_summary = st.session_state.ref_summary_random
    else:
        st.info("Clique em 'Gerar Artigo Aleatório' para carregar um artigo da base.")
elif option == "Colar manualmente":
    article = st.sidebar.text_area("Cole seu artigo aqui:", height=200)
    reference_summary = st.sidebar.text_area("Resumo humano (opcional):", height=100)

##############################################################################
# 5) Mostrar Artigo e Resumo Humano
##############################################################################
st.subheader("Texto Original")
if article.strip():
    st.write(article)
else:
    st.info("Nenhum artigo disponível no momento.")

if reference_summary.strip():
    st.markdown("**Resumo Humano (Referência):**")
    st.write(reference_summary)
else:
    st.warning("Resumo humano não fornecido ou está vazio. ROUGE pode ficar zerado.")

st.write("---")

##############################################################################
# 6) Botão para Gerar Resumos e Comparar
##############################################################################
if st.button("Gerar Resumos e Comparar"):
    if not article.strip():
        st.error("Não há artigo para resumir.")
        st.stop()

    # Dicionário de resumos
    generated_summaries = {}

    # BART Original
    bart_model = BartSummarizer()
    start_time = time.time()
    bart_summary = bart_model.summarize(article)
    bart_time = time.time() - start_time
    generated_summaries["BART"] = bart_summary

    # GPT-3.5 Turbo
    gpt_model = GPTSummarizer(os.getenv("OPENAI_API_KEY"))
    start_time = time.time()
    gpt_summary = gpt_model.summarize(article)
    gpt_time = time.time() - start_time
    generated_summaries["GPT-3.5 Turbo"] = gpt_summary

    # LLaMA (Ollama)
    llama_model = LlamaSummarizer()
    start_time = time.time()
    llama_summary = llama_model.summarize(article)
    llama_time = time.time() - start_time
    generated_summaries["LLaMA"] = llama_summary

    # Deepseek-r1 Local
    deepseek_local_model = DeepseekSummarizer()
    start_time = time.time()
    deepseek_loca_summary = deepseek_local_model.summarize(article)
    deepseek_local_time = time.time() - start_time
    generated_summaries["DeepSeek-r1"] = deepseek_loca_summary

    # BART Fine-Tuned
    bart_finetuned_model = BartFineTunedSummarizer()
    start_time = time.time()
    bart_finetuned_summary = bart_finetuned_model.summarize(article)
    bart_finetuned_time = time.time() - start_time
    generated_summaries["BART Fine-Tuned"] = bart_finetuned_summary

    # Calcular ROUGE
    if reference_summary.strip():
        evaluation_results = evaluate_summary(generated_summaries, reference_summary)
    else:
        evaluation_results = {}

    # Montar estrutura
    rows = []
    time_data = {}
    rouge1_data = {}
    rouge2_data = {}
    rougel_data = {}

    for model_name, summary_text in generated_summaries.items():
        tempo = 0.0
        if model_name == "BART":
            tempo = bart_time
        elif model_name == "GPT-3.5 Turbo":
            tempo = gpt_time
        elif model_name == "LLaMA":
            tempo = llama_time
        elif model_name == "DeepSeek-r1":
            tempo = deepseek_local_time
        elif model_name == "BART Fine-Tuned":
            tempo = bart_finetuned_time

        time_data[model_name] = tempo

        rouge_scores = evaluation_results.get(model_name, {})
        r1 = round(rouge_scores.get("ROUGE-1", 0.0), 4)
        r2 = round(rouge_scores.get("ROUGE-2", 0.0), 4)
        rl = round(rouge_scores.get("ROUGE-L", 0.0), 4)

        rouge1_data[model_name] = r1
        rouge2_data[model_name] = r2
        rougel_data[model_name] = rl

        rows.append({
            "Modelo": model_name,
            "Tempo (s)": f"{tempo:.2f}",
            "Resumo": summary_text,
            "ROUGE-1": f"{r1:.4f}",
            "ROUGE-2": f"{r2:.4f}",
            "ROUGE-L": f"{rl:.4f}"
        })

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Tabela de Comparação (DataFrame Reordenável)
    # ---------------------------------------------------------
    st.subheader("Tabela de Comparação")

    st.markdown("""
    <style>
    [data-testid="stDataFrame"] {
        white-space: pre-wrap;
        word-wrap: break-word;
        height: 650px !important;
    }
    .tooltip {
      position: relative;
      display: inline-block;
      border-bottom: 1px dotted #555;
    }
    .tooltip .tooltiptext {
      visibility: hidden;
      width: 320px;
      background-color: #555;
      color: #fff;
      text-align: left;
      border-radius: 6px;
      padding: 5px 8px;
      position: absolute;
      z-index: 1;
      bottom: 125%;
      margin-left: -80px;
      opacity: 0;
      transition: opacity 0.3s;
      font-size: 0.90rem;
    }
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p>
      Passe o mouse nas colunas 
      <span class="tooltip">ROUGE-1
        <span class="tooltiptext">
          <b>ROUGE-1:</b> Mede quantas palavras (unigramas) do seu resumo
          coincidem com as do resumo humano.<br><br>
          Ótimo para checar semelhança lexical básica.
        </span>
      </span>, 
      <span class="tooltip">ROUGE-2
        <span class="tooltiptext">
          <b>ROUGE-2:</b> Mede quantas sequências de 2 palavras (bigramas)
          aparecem em ambos os resumos.<br><br>
          Indica coerência entre frases.
        </span>
      </span>, 
      <span class="tooltip">ROUGE-L
        <span class="tooltiptext">
          <b>ROUGE-L:</b> Mede a maior subsequência comum (LCS).
          <br><br>Ajuda a verificar a ordem e coesão das sentenças.
        </span>
      </span>.
    </p>
    """, unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True)

    # -----------------------------------------------------------
    # Tabela de Traduções (HTML custom)
    # -----------------------------------------------------------
    st.subheader("Tradução dos Resumos para Português")

    # Gerar lista translations
    translations = []
    if reference_summary.strip():
        translations.append({
            "Modelo": "Resumo Humano (PT-BR)",
            "Resumo (PT)": translate_to_pt(reference_summary)
        })

    for row_data in rows:
        model_name = row_data["Modelo"]
        summary_en = row_data["Resumo"]
        translated_txt = translate_to_pt(summary_en)
        translations.append({
            "Modelo": f"{model_name} (PT-BR)",
            "Resumo (PT)": translated_txt
        })

    # Montar HTML custom
    table_html = "<style>"
    table_html += """
.custom-translation-table {
  width: 100%;
  table-layout: auto;
  border-collapse: collapse;
}
.custom-translation-table th, .custom-translation-table td {
  border: 1px solid #ddd;
  padding: 8px;
  vertical-align: top;
  text-align: left;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.custom-translation-table th {
  background-color: #f2f2f2;
  width: 15%;
}
"""
    table_html += "</style>"
    table_html += "<table class='custom-translation-table'>"
    table_html += "<tr><th>Modelo</th><th>Resumo (PT)</th></tr>"

    for trans_row in translations:
        modelo = trans_row["Modelo"]
        resumo_pt = trans_row["Resumo (PT)"]
        table_html += f"<tr><td>{modelo}</td><td>{resumo_pt}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    # -----------------------------------------------------------
    # Gráficos no final
    # -----------------------------------------------------------
    st.write("---")
    st.subheader("Gráficos de Comparação")

    # Gráfico de tempo
    st.subheader("Tempo de Execução")
    time_df = pd.DataFrame({"Tempo (s)": time_data})
    st.bar_chart(time_df)

    # Gráficos de ROUGE (apenas se houver resumo humano)
    if reference_summary.strip():
        st.subheader("Comparação ROUGE-1")
        r1_df = pd.DataFrame({"ROUGE-1": rouge1_data})
        st.bar_chart(r1_df)

        st.subheader("Comparação ROUGE-2")
        r2_df = pd.DataFrame({"ROUGE-2": rouge2_data})
        st.bar_chart(r2_df)

        st.subheader("Comparação ROUGE-L")
        rl_df = pd.DataFrame({"ROUGE-L": rougel_data})
        st.bar_chart(rl_df)
    else:
        st.info("Não é possível gerar gráficos de ROUGE pois não há resumo humano.")

else:
    st.info("Selecione a fonte do artigo, gere artigo aleatório (se quiser) e clique em 'Gerar Resumos e Comparar'.")
