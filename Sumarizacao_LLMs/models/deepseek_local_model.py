import ollama
import re

class DeepseekSummarizer:
    def __init__(self, model_name="deepseek-r1"):
        self.model_name = model_name

    def summarize(self, text):
        prompt = f"Summarize the following text in one paragraph: {text}"
        resposta = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        summary = resposta["message"]["content"]

        # 🔍 Remover texto entre <think> e </think>
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

        return summary

# Teste com um texto de exemplo
if __name__ == "__main__":
    model = DeepseekSummarizer()
    texto_teste = "A OpenAI lançou o ChatGPT para facilitar a interação com modelos de linguagem. Ele permite responder perguntas, gerar textos e muito mais."
    print("Resumo gerado com deepseek-r1:", model.summarize(texto_teste))
