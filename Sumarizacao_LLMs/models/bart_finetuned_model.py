import torch
from transformers import BartTokenizer, BartForConditionalGeneration

import torch
print("CUDA disponível:", torch.cuda.is_available())
print("Dispositivo em uso:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class BartFineTunedSummarizer:
    def __init__(self):
        """Carrega o modelo BART treinado no CNN/DailyMail"""
        self.model_path = "./bart_finetuned"  # Caminho local do modelo treinado
        self.tokenizer = BartTokenizer.from_pretrained(self.model_path)

        # Define se vai rodar na GPU (se disponível) ou na CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carrega o modelo na GPU, se disponível
        self.model = BartForConditionalGeneration.from_pretrained(self.model_path).to(self.device)

    def summarize(self, text, max_length=100, min_length=50):
        """Gera um resumo do texto fornecido"""
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        summary_ids = self.model.generate(inputs.input_ids, max_length=max_length, min_length=min_length, num_beams=4)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Teste do modelo fine-tuned na GPU
if __name__ == "__main__":
    model = BartFineTunedSummarizer()
    texto_teste = "O ChatGPT foi desenvolvido pela OpenAI para gerar textos de forma inteligente, respondendo perguntas e interagindo com usuários."
    print("Resumo Fine-Tuned:", model.summarize(texto_teste))
