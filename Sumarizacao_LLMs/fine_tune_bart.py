import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# 📌 Carregar dataset JSONL convertido
dataset = load_dataset("json", data_files={
    "train": "data/cnn_dailymail/jsonl/train.jsonl",
    "validation": "data/cnn_dailymail/jsonl/validation.jsonl"
})

# 📌 Carregar modelo BART alternativo e tokenizer
model_name = "facebook/bart-base"  # Versão base para fine-tuning
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 📌 Verificar se há GPU disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📌 Treinamento rodando em: {device.upper()}")
model.to(device)

# 📌 Tokenizar os dados
def preprocess_data(examples):
    inputs = tokenizer(examples["article"], max_length=1024, truncation=True, padding="max_length")
    targets = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# 📌 Definir parâmetros de treinamento
training_args = TrainingArguments(
    output_dir="bart_finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # Ajustado para GPU
    per_device_eval_batch_size=4,  # Ajustado para GPU
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="logs",
    report_to="none"
)

# 📌 Criar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

# 📌 Iniciar treinamento
trainer.train()

# 📌 Salvar modelo treinado
model.save_pretrained("bart_finetuned")
tokenizer.save_pretrained("bart_finetuned")

print("✅ Fine-Tuning concluído e modelo salvo em bart_finetuned/")
