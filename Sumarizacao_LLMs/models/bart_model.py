from transformers import BartTokenizer, BartForConditionalGeneration

class BartSummarizer:
    def __init__(self):
        self.model_name = "facebook/bart-base"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name, forced_bos_token_id=0)

    def summarize(self, text, max_length=100, min_length=50):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs.input_ids, max_length=max_length, min_length=min_length, num_beams=4)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Teste
if __name__ == "__main__":
    model = BartSummarizer()
    text = "Seu texto longo aqui..."
    print("Resumo:", model.summarize(text))
