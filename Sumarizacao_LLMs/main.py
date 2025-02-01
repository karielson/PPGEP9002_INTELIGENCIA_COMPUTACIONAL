from models.bart_model import BartSummarizer
from models.gpt_model import GPTSummarizer
from models.llama_model import LlamaSummarizer
from models.deepseek_local_model import DeepseekSummarizer
from models.deepseek_model import DeepSeekSummarizer
from evaluate_rouge import evaluate_summary
import pandas as pd
import os
from models.bart_finetuned_model import BartFineTunedSummarizer  # Modelo Fine-Tuned
import time

# Carregar chaves de API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("A chave da API OpenAI não foi encontrada. Defina a variável de ambiente OPENAI_API_KEY.")

# deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
# if not deepseek_api_key:
#     raise ValueError("A chave da API DeepSeek não foi encontrada. Defina a variável de ambiente DEEPSEEK_API_KEY.")

# Carregar uma amostra do dataset
df = pd.read_csv("data/cnn_dailymail/test.csv").sample(1)
article = df["article"].values[0]
reference_summary = df["highlights"].values[0]

# Exibir o texto original
print("\n📌 Texto original:", article[:500], "...\n")
print("🎯 Resumo humano:", reference_summary, "\n")

# Lista de modelos e seus tempos de resposta
generated_summaries = {}

# Teste com BART
bart_model = BartSummarizer()
start_time = time.time()
bart_summary = bart_model.summarize(article)
bart_time = time.time() - start_time
generated_summaries["BART"] = bart_summary

print(f"🔹 BART: {bart_summary}")
print(f"⏱ Tempo de resposta: {bart_time:.2f} segundos\n")

# Teste com GPT-3.5 Turbo
gpt_model = GPTSummarizer(api_key)
start_time = time.time()
gpt_summary = gpt_model.summarize(article)
gpt_time = time.time() - start_time
generated_summaries["GPT-3.5 Turbo"] = gpt_summary

print(f"🔹 GPT-3.5 Turbo: {gpt_summary}")
print(f"⏱ Tempo de resposta: {gpt_time:.2f} segundos\n")

# Teste com LLaMA 3 via Ollama
llama_model = LlamaSummarizer()
start_time = time.time()
llama_summary = llama_model.summarize(article)
llama_time = time.time() - start_time
generated_summaries["LLaMA"] = llama_summary

print(f"🔹 LLaMA 3 (Ollama): {llama_summary}")
print(f"⏱ Tempo de resposta: {llama_time:.2f} segundos\n")

# # Teste com DeepSeek API
# deepseek_model = DeepSeekSummarizer(deepseek_api_key)
# start_time = time.time()
# deepseek_summary, deepseek_time = deepseek_model.summarize(article)
# deepseek_time = time.time() - start_time
# generated_summaries["DeepSeek"] = deepseek_summary

# print(f"🔹 DeepSeek: {deepseek_summary}")
# print(f"⏱ Tempo de resposta: {deepseek_time:.2f} segundos\n")

# Teste com deepseek-r1 via Ollama
deepseek_local_model = DeepseekSummarizer()
start_time = time.time()
deepseek_local_summary = deepseek_local_model.summarize(article)  # Agora retorna sem <think>...</think>
deepseek_local_time = time.time() - start_time
generated_summaries["DeepSeek-r1 (Ollama)"] = deepseek_local_summary

print(f"🔹 DeepSeek-r1 (Ollama): {deepseek_local_summary}")
print(f"⏱ Tempo de resposta: {deepseek_local_time:.2f} segundos\n")



# Teste com o BART Fine-Tuned
bart_finetuned_model = BartFineTunedSummarizer()
start_time = time.time()
bart_finetuned_summary = bart_finetuned_model.summarize(article)
bart_finetuned_time = time.time() - start_time
generated_summaries["BART Fine-Tuned:"] = bart_finetuned_summary

print(f"🔹 BART Fine-Tuned: {bart_finetuned_summary}")
print(f"⏱ Tempo de resposta: {bart_finetuned_time:.2f} segundos\n")



# Avaliação com ROUGE
print("\n🔍 Avaliação ROUGE:")
evaluation_results = evaluate_summary(generated_summaries, reference_summary)



for model, scores in evaluation_results.items():
    print(f"\n📊 ROUGE - {model}:")
    
    # Evitar erro KeyError verificando se as chaves existem antes de acessar
    rouge1 = scores.get("ROUGE-1", 0.0)
    rouge2 = scores.get("ROUGE-2", 0.0)
    rougeL = scores.get("ROUGE-L", 0.0)

    print(f"  ROUGE-1: {rouge1:.4f}")
    print(f"  ROUGE-2: {rouge2:.4f}")
    print(f"  ROUGE-L: {rougeL:.4f}")


