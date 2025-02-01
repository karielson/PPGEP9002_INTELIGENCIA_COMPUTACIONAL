from datasets import load_dataset
import pandas as pd
import os

def load_and_save_dataset():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    os.makedirs("data/cnn_dailymail", exist_ok=True)
    
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split])
        df.to_csv(f"data/cnn_dailymail/{split}.csv", index=False)

    print("Dataset salvo em data/cnn_dailymail/")

if __name__ == "__main__":
    load_and_save_dataset()
