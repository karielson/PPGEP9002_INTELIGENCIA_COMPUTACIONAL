import os
from openai import OpenAI

class GPTSummarizer:
    def __init__(self, api_key):
        """Inicializa o cliente OpenAI com a chave da API."""
        self.client = OpenAI(api_key=api_key)

    def summarize(self, text, max_tokens=100):
        """Gera um resumo do texto fornecido usando o modelo GPT-3.5-turbo."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": f"Summarize the following text in one paragraph: {text}"}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content  # Retorna o resumo gerado

# Teste do código
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")  # Obtém a chave da API da variável de ambiente
    if not api_key:
        raise ValueError("A chave da API OpenAI não foi encontrada. Defina a variável de ambiente OPENAI_API_KEY.")

    summarizer = GPTSummarizer(api_key)
    text = "A inteligência artificial está transformando diversas indústrias, permitindo automação, análise de dados avançada e novas formas de interação entre humanos e máquinas."
    
    resumo = summarizer.summarize(text)
    print("Resumo:", resumo)