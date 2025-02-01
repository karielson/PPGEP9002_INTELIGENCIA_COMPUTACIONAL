import evaluate

# Carregar a métrica ROUGE
rouge = evaluate.load("rouge")

def evaluate_summary(predicted_summaries, reference_summary):
    """
    Avalia os resumos gerados comparando-os com o resumo humano de referência.
    """
    scores = {}
    for model_name, predicted in predicted_summaries.items():
        result = rouge.compute(predictions=[predicted], references=[reference_summary])

        # 🔍 Debug: Ver a saída exata da métrica ROUGE
        # print(f"\n🔎 Debug - Resultado ROUGE para {model_name}: {result}")

        # Verificar se 'rouge1' está no resultado antes de acessar
        rouge1_score = result.get("rouge1", 0.0)
        rouge2_score = result.get("rouge2", 0.0)
        rougeL_score = result.get("rougeL", 0.0)

        # Se os valores forem dicionários, pegar 'fmeasure'
        if isinstance(rouge1_score, dict):
            rouge1_score = rouge1_score.get("fmeasure", 0.0)
        if isinstance(rouge2_score, dict):
            rouge2_score = rouge2_score.get("fmeasure", 0.0)
        if isinstance(rougeL_score, dict):
            rougeL_score = rougeL_score.get("fmeasure", 0.0)

        # Armazenar os resultados
        scores[model_name] = {
            "ROUGE-1": rouge1_score,
            "ROUGE-2": rouge2_score,
            "ROUGE-L": rougeL_score,
        }

    return scores


# Teste da função
if __name__ == "__main__":
    reference_summary = "O ChatGPT foi lançado pela OpenAI para facilitar a interação com IA."

    predicted_summaries = {
        "BART": "A OpenAI lançou o ChatGPT para responder perguntas de usuários.",
        "GPT-3.5 Turbo": "ChatGPT, criado pela OpenAI, permite responder perguntas.",
        "LLaMA": "OpenAI lançou o ChatGPT para interação com IA e resposta de perguntas.",
        "DeepSeek": "O ChatGPT, da OpenAI, aprimora a comunicação com inteligência artificial."
    }

    evaluation_results = evaluate_summary(predicted_summaries, reference_summary)

    for model, scores in evaluation_results.items():
        print(f"\n🔍 ROUGE - {model}:")
        print(f"  ROUGE-1: {scores['ROUGE-1']:.4f}")
        print(f"  ROUGE-2: {scores['ROUGE-2']:.4f}")
        print(f"  ROUGE-L: {scores['ROUGE-L']:.4f}")
