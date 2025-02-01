import pandas as pd
import json
import os

# Criar diretório JSONL
os.makedirs("data/cnn_dailymail/jsonl", exist_ok=True)

# Converte CSV para JSONL
def convert_csv_to_jsonl(split):
    df = pd.read_csv(f"data/cnn_dailymail/{split}.csv")

    jsonl_path = f"data/cnn_dailymail/jsonl/{split}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json.dump({"article": row["article"], "summary": row["highlights"]}, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ {split}.csv convertido para JSONL!")

# Converter todos os splits
for split in ["train", "validation", "test"]:
    convert_csv_to_jsonl(split)
