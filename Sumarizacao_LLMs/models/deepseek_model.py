import os
import time
from openai import OpenAI

class DeepSeekSummarizer:
    def __init__(self, api_key=None):
        """Inicializa o cliente DeepSeek com a chave da API."""
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("A chave da API DeepSeek não foi encontrada. Defina a variável de ambiente DEEPSEEK_API_KEY.")
        
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")

    def summarize(self, text, max_tokens=100):
        """Gera um resumo do texto fornecido usando o modelo DeepSeek e mede o tempo de resposta."""
        start_time = time.time()

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": f"Summarize the following text in one paragraph: {text}"}
            ],
            max_tokens=max_tokens
        )
        end_time = time.time()
        execution_time = end_time - start_time

        return response.choices[0].message.content, execution_time  # Retorna o resumo e o tempo gasto

# Teste do código
if __name__ == "__main__":
    summarizer = DeepSeekSummarizer()
    text = "A inteligência artificial está transformando diversas indústrias, permitindo automação, análise de dados avançada e novas formas de interação entre humanos e máquinas."
    
    resumo, tempo = summarizer.summarize(text)
    print("Resumo:", resumo)
    print(f"⏱ Tempo de resposta: {tempo:.2f} segundos")
