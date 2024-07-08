from nltk import sent_tokenize, word_tokenize
from spellchecker import SpellChecker
import language_tool_python
import pandas as pd
import torch
import evaluate
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


spell = SpellChecker()
tool = language_tool_python.LanguageTool('en-US')

def remove_redundant_phrases(summary):
    """
    Removes redundant sentences from a summary.
    """
    sentences = sent_tokenize(summary)
    unique_sentences = list(dict.fromkeys(sentences))  # Remove duplicates while preserving order
    return ' '.join(unique_sentences)


def correct_grammar_and_spelling(summary):
    """
    Corrects grammar and spelling errors in a summary.
    """
    words = word_tokenize(summary)
    corrected_words = [spell.correction(word) if word.lower() in spell else word for word in words]
    corrected_summary = ' '.join(corrected_words)
    corrected_summary = tool.correct(corrected_summary)  # Use LanguageTool for grammar correction
    return corrected_summary


def trim_unnecessary_information(summary):
    """
    Trims unnecessary phrases like 'In conclusion', 'To summarize', 'In summary' from a summary.
    """
    fillers = ["In conclusion", "To summarize", "In summary"]
    for filler in fillers:
        summary = summary.replace(filler, '')
    return summary.strip()


def post_process_summary(summary):
    """
    Post-processes a summary by removing redundancy, correcting grammar and spelling, and trimming unnecessary information.
    """
    summary = remove_redundant_phrases(summary)
    summary = correct_grammar_and_spelling(summary)
    summary = trim_unnecessary_information(summary)
    return summary


def generate_summary(model, tokenizer, input_ids):
    """
    Generates a summary using a sequence-to-sequence model.
    """
    is_t5_model = hasattr(model.config, 'decoder_start_token_id')
    with torch.no_grad():
        if is_t5_model:
            outputs = model.generate(input_ids=input_ids, 
                                     max_length=100,
                                     temperature=0.2,
                                     num_beams=1,
                                     do_sample=True)
            text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            normalized_logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
            if torch.isnan(normalized_logits).any() or torch.isinf(normalized_logits).any():
                normalized_logits[torch.isnan(normalized_logits) | torch.isinf(normalized_logits)] = -1e9
            probs = torch.softmax(normalized_logits, dim=-1)
            if (probs < 0).any() or (probs > 1).any() or torch.isnan(probs).any() or torch.isinf(probs).any():
                raise ValueError("Invalid probabilities encountered during sampling")
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            text_output = tokenizer.decode(next_tokens.tolist(), skip_special_tokens=True)
    return text_output


def summarize_conversations(t5_original_model, t5_original_tokenizer, 
                            google_t5_original_model, google_t5_original_tokenizer,
                            peft_basic_model, peft_complex_model,
                            t5_2_model, t5_2_tokenizer,
                            google_1_model, google_1_tokenizer,
                            inputs, human_baseline_summaries, prompt_template):
    """
    Summarizes conversations using multiple models and compares with human baseline summaries.
    """
    t5_original_summaries = []
    google_original_summaries = []
    peft_basic_summaries = []
    peft_complex_summaries = []
    t5_2_model_summaries = []
    google_1_model_summaries = []

    for input_text in inputs:
        prompt = prompt_template.format(input=input_text)
        input_ids = t5_original_tokenizer(prompt, return_tensors="pt").input_ids

        original_summary = generate_summary(t5_original_model, t5_original_tokenizer, input_ids)
        google_summary = generate_summary(google_t5_original_model, google_t5_original_tokenizer, input_ids)
        peft_basic_summary = generate_summary(peft_basic_model, google_t5_original_tokenizer, input_ids)
        peft_complex_summary = generate_summary(peft_complex_model, google_t5_original_tokenizer, input_ids)
        t5_2_model_summary = generate_summary(t5_2_model, t5_2_tokenizer, input_ids)
        google_1_model_summary = generate_summary(google_1_model, google_1_tokenizer, input_ids)

        t5_original_summaries.append(post_process_summary(original_summary))
        google_original_summaries.append(post_process_summary(google_summary))
        peft_basic_summaries.append(post_process_summary(peft_basic_summary))
        peft_complex_summaries.append(post_process_summary(peft_complex_summary))
        t5_2_model_summaries.append(post_process_summary(t5_2_model_summary))
        google_1_model_summaries.append(post_process_summary(google_1_model_summary))

    zipped_summaries = list(zip(human_baseline_summaries, t5_original_summaries, google_original_summaries, 
                                peft_basic_summaries, peft_complex_summaries, t5_2_model_summaries, google_1_model_summaries))
    
    df = pd.DataFrame(zipped_summaries, columns=["human_baseline_summaries", "t5_original_summaries", "google_original_summaries", 
                                                 "peft_basic_summaries", "peft_complex_summaries", "t5_2_model_summaries", 
                                                 "google_1_model_summaries"])
    
    return df


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


def compute_rouge_scores(t5_original_summaries, google_original_summaries, 
                         peft_basic_summaries, peft_complex_summaries,
                         t5_2_model_summaries, google_1_model_summaries,
                         human_baseline_summaries):
    """
    Computes ROUGE scores for summaries generated by different models.
    """
    # Compute ROUGE scores for each model's summaries
    original_model_results = compute_rouge_for_model(t5_original_summaries, human_baseline_summaries)
    google_model_results = compute_rouge_for_model(google_original_summaries, human_baseline_summaries)
    peft_basic_results = compute_rouge_for_model(peft_basic_summaries, human_baseline_summaries)
    peft_complex_results = compute_rouge_for_model(peft_complex_summaries, human_baseline_summaries)
    t5_2_model_results = compute_rouge_for_model(t5_2_model_summaries, human_baseline_summaries)
    google_1_model_results = compute_rouge_for_model(google_1_model_summaries, human_baseline_summaries)

    return original_model_results, google_model_results, peft_basic_results, peft_complex_results, \
           t5_2_model_results, google_1_model_results


def compute_rouge_scores_for_models(summary_df):
    """
    Computes and prints ROUGE scores for summaries generated by different models.
    """
    # Compute ROUGE scores for each model's summaries
    original_model_results, google_model_results, peft_basic_results, peft_complex_results, \
    t5_2_model_results, google_1_model_results = compute_rouge_scores(
        summary_df['t5_original_summaries'], summary_df['google_original_summaries'], 
        summary_df['peft_basic_summaries'], summary_df['peft_complex_summaries'], 
        summary_df['t5_2_model_summaries'], summary_df['google_1_model_summaries'], 
        summary_df['human_baseline_summaries']
    )

    def print_results(model_name, results):
        print(f'{model_name} RESULTS:')
        for key, value in results.items():
            print(f"{key}: {value:.2%}")
        print()

    # Print ROUGE scores for each model
    print_results('ORIGINAL MODEL', original_model_results)
    print_results('GOOGLE MODEL', google_model_results)
    print_results('PEFT BASIC MODEL', peft_basic_results)
    print_results('PEFT COMPLEX MODEL', peft_complex_results)
    print_results('FINETUNED T5 BCN DIALOG SUM MODEL', t5_2_model_results)
    print_results('FINETUNED T5 BCN GOOGLE MODEL', google_1_model_results)

    return original_model_results, google_model_results, peft_basic_results, peft_complex_results, \
           t5_2_model_results, google_1_model_results


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1 - amount) with the input color
    and adding the amount to each of the RGB channels.
    """
    try:
        c = to_rgba(color)
    except ValueError:
        c = to_rgba(color, alpha=1.0)
    c = [c[0] * (1 - amount) + amount, c[1] * (1 - amount) + amount, c[2] * (1 - amount) + amount, c[3]]
    return c


def plot_rouge_accuracies_and_improvements(model_names, rouge_scores, baseline_scores):
    """
    Plots ROUGE-1 and ROUGE-L (ROUGE-LCS) accuracy scores and their improvements over baseline
    for each model in two separate plots side by side.
    """
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    rouge_types = ['rouge1', 'rougeL']
    for idx, rouge_type in enumerate(rouge_types):
        ax = axes[idx]
        scores = [score[rouge_type] * 100 for score in rouge_scores]

        max_score = max(scores)

        colors = [lighten_color('green', amount=1 - (score / max_score)) for score in scores]

        bars = ax.bar(model_names, scores, color=colors)
        ax.set_xlabel('Models')
        ax.set_ylabel(f'{rouge_type.upper()} Score (%)')
        ax.set_title(f'{rouge_type.upper()} Scores for Models')

        ax.set_xticks(range(len(model_names)))  # Setting the number of ticks
        ax.set_xticklabels(model_names, rotation=45)

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.2f}%', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    for idx, rouge_type in enumerate(rouge_types):
        ax = axes[idx + 2]
        baseline_score = baseline_scores.get(rouge_type, 0)
        scores = [score.get(rouge_type, float('nan')) for score in rouge_scores]

        # Calculate improvements relative to baseline
        improvements = []
        for score in scores:
            if baseline_score == 0:
                improvement = score  # Directly set improvement to 0% if baseline is 0
            elif baseline_score is None or score is None:
                improvement = float('nan')
            else:
                improvement = (score - baseline_score) / baseline_score * 100
            improvements.append(improvement)

        bars = ax.bar(model_names, improvements, color='skyblue')
        ax.axhline(0, color='red', linewidth=1, linestyle='--')
        ax.set_xlabel('Models')
        ax.set_ylabel(f'Improvement in {rouge_type.upper()} (%)')
        ax.set_title(f'Improvement in {rouge_type.upper()} over Baseline')

        ax.set_xticks(range(len(model_names)))  # Setting the number of ticks
        ax.set_xticklabels(model_names, rotation=45)

        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'{improvement:.2f}%', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.suptitle('ROUGE Scores and Improvements', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
