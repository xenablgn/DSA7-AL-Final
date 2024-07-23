import evaluate
import pandas as pd
from text_processing import generate_summary, post_process_summary


def compute_rouge_for_model(predictions, references):
    """
    Computes ROUGE scores between predicted summaries and reference summaries.
    """
    rouge = evaluate.load('rouge')
    results = rouge.compute(
        predictions=predictions,
        references=references[:len(predictions)],  # Ensure references match predictions length
        use_aggregator=True,
        use_stemmer=True,
    )
    return results


def summarize_and_evaluate(model, tokenizer, input_texts, human_baseline_summaries, model_type, prompt_template):
    """
    Summarizes texts using a single model and evaluates the summaries.
    """
    summaries = []
    for input_text in input_texts:

        formatted_prompt = prompt_template.format(input=input_text)
        input_text_only = formatted_prompt.split('Input:')[1].strip() if 'Input:' in formatted_prompt else formatted_prompt

        try:
            summary = generate_summary(model, tokenizer, input_text_only, model_type)
            summaries.append(post_process_summary(summary))
        except Exception as e:
            summaries.append(f"An error occurred: {str(e)}")

    if human_baseline_summaries is None:
        human_baseline_summaries = [""] * len(input_texts)
        
    elif len(human_baseline_summaries) < len(input_texts):
        human_baseline_summaries.extend([""] * (len(input_texts) - len(human_baseline_summaries)))

    df = pd.DataFrame({
        "Input Text": input_texts,
        "Human Baseline Summaries": human_baseline_summaries,
        "Model Summaries": summaries
    })

    rouge_scores = {}
    if human_baseline_summaries:
        rouge_scores = compute_rouge_for_model(summaries, human_baseline_summaries)

    return df, rouge_scores

