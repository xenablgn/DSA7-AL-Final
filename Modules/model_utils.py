import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

def cache_directory(model_name):
    """
    Create a cache directory based on the model name.
    """
    cache_dir = os.path.join("cache", model_name.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory created: {cache_dir}")
    return cache_dir


def load_model_and_tokenizer(model_type, version=None, torch_dtype=torch.bfloat16, is_trainable=False):
    """
    Load model and tokenizer based on the given parameters with caching.
    """
    if model_type == 'Google-Base':
        model_name = "google/flan-t5-base"
        cache_dir = cache_directory(model_name)
        print(f"Loading Google Base model from cache: {cache_dir}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    elif model_type == 'Google-Peft-Lora':
        model_name = "google/flan-t5-base"
        peft_path = "E:/Epita/action_learning/DSA7-AL-Final-dev/Models/Google-Peft-Lora-COMPLEX"
        cache_dir = cache_directory(model_name)
        print(f"Loading base model for PEFT from cache: {cache_dir}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype, cache_dir=cache_dir)
        print(f"Loading PEFT model from: {peft_path}")
        model = PeftModel.from_pretrained(base_model, peft_path, torch_dtype=torch_dtype, is_trainable=is_trainable)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    elif model_type == 'T5-Double-Tuned':
        model_path = "E:/Epita/action_learning/DSA7-AL-Final-dev/Models/T52-BCN-DIALOG/checkpoint-3115"
        tokenizer_path = "E:/Epita/action_learning/DSA7-AL-Final-dev/Models/T52-BCN-DIALOG"
        cache_dir = cache_directory("T5-Double-Tuned")
        print(f"Loading T5 Double Tuned model from cache: {cache_dir}")
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch_dtype, cache_dir=cache_dir)
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)
    
    elif model_type == 'Google-Fine-Tuned':
        model_name = "google/flan-t5-base"
        model_path = "E:/Epita/action_learning/DSA7-AL-Final-dev/Models/GOOGLE-BCN/checkpoint-2225"
        tokenizer_path = "E:/Epita/action_learning/DSA7-AL-Final-dev/Models/GOOGLE-BCN"
        cache_dir = cache_directory("Google-Fine-Tuned")
        print(f"Loading Google Fine Tuned model from cache: {cache_dir}")
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch_dtype, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    else:
        raise ValueError("Unsupported model type. Use 'Google-Base', 'Google-Peft-Lora', 'T5-Double-Tuned', or 'Google-Fine-Tuned'.")

    return model, tokenizer
