import ollama

class LlamaSummarizer:
    def __init__(self, model_name="llama3", max_tokens=100):
        self.model_name = model_name

    def summarize(self, text):
        prompt = f"Summarize the following text in one paragraph: {text}"
        resposta = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return resposta["message"]["content"]

# Teste com um texto de exemplo
if __name__ == "__main__":
    model = LlamaSummarizer()
    texto_teste = "A OpenAI lançou o ChatGPT para facilitar a interação com modelos de linguagem. Ele permite responder perguntas, gerar textos e muito mais."
    print("Resumo gerado com LLaMA 3:", model.summarize(texto_teste))
