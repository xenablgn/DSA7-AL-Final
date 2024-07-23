import re
import pandas as pd
import torch
from nltk import sent_tokenize, word_tokenize
from spellchecker import SpellChecker
import language_tool_python
import evaluate
from utils import MAX_SEQUENCE_LENGTHS

spell = SpellChecker()
tool = language_tool_python.LanguageTool('en-US')

def clean_text(text):
    """
    Cleans the text by removing unwanted characters and extra spaces.
    """
    text = ' '.join(text.split())
    
    text = re.sub(r'[^\w\s.!?]', '', text)
    return text


def chunk_text(text, tokenizer, max_length):
    """
    Splits the text into chunks that fit within the model's token limit.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    token_ids = tokens.input_ids[0]
    chunks = []
    for i in range(0, len(token_ids), max_length):
        chunk = token_ids[i:i + max_length]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks


def generate_summary(model, tokenizer, input_text, model_type, max_length=100, temperature=0.2, num_beams=1, num_return_sequences=1, top_p=0.95):
    """
    Generates a summary using the provided model and tokenizer based on model type.
    Processes long input texts by chunking them if necessary.
    """
    
    max_sequence_length = MAX_SEQUENCE_LENGTHS.get(model_type, 512)  # Default to 512 if model_type is not found
    cleaned_input = clean_text(input_text)
    
    chunks = chunk_text(cleaned_input, tokenizer, max_sequence_length)

    summaries = []
    for chunk in chunks:
        input_ids = tokenizer(chunk, return_tensors="pt").input_ids

        if model_type in ['Google-Base', 'Google-Peft-Lora', 'T5-Double-Tuned', 'Google-Fine-Tuned']:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=num_beams,
                    do_sample=True
                )
            chunk_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        else:
            raise ValueError("Unsupported model type.")

        summaries.append(chunk_summary)

    full_summary = " ".join(summaries)
    
    return full_summary



def remove_redundant_phrases(summary):
    """
    Removes redundant sentences from a summary.
    """
    sentences = sent_tokenize(summary)
    seen_sentences = set()
    unique_sentences = []
    for sentence in sentences:
        if sentence not in seen_sentences:
            unique_sentences.append(sentence)
            seen_sentences.add(sentence)
    return ' '.join(unique_sentences)


def correct_grammar_and_spelling(summary):
    """
    Corrects grammar and spelling errors in a summary.
    """
    words = word_tokenize(summary)
    corrected_words = [spell.correction(word) if word.lower() in spell else word for word in words]
    corrected_summary = ' '.join(corrected_words)
    corrected_summary = tool.correct(corrected_summary)  
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

