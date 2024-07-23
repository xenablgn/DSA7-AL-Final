# Full-Fine Tuning and PEFT Tuning for T5 Models

**Specialization: Data Science and Analytics**

---

## Overview

This repository demonstrates the application of full-fine tuning and parameter-efficient fine-tuning (PEFT) for various T5 models, including Microsoft-phi2, T5.1, T5.2, Google T5 Flan, and GPT-2. It includes methods for dialogue summarization and adaptive summary generation. The repository also covers the deployment of these models using FastAPI and Streamlit.

---

## Zero-Shot Learning for Text Summarization

### Improving Google FLAN's Performance

The objective is to enhance the performance of Google FLAN through PEFT fine-tuning for adaptive summary generation and classification. Zero-shot learning capabilities are explored to understand model performance on tasks it hasn't been explicitly trained on.

---

## PEFT Tuning Example Notebook for Microsoft-phi2

### Overview

This notebook demonstrates parameter-efficient fine-tuning (PEFT) for the Microsoft-phi2 model using the `neil-code/dialogsum-test` dataset. It focuses on dialogue summarization tasks and evaluates performance using the ROUGE score. Inference is tested on the `abisee/cnn_dailymail` dataset to assess zero-shot learning capabilities.

### Datasets

- **neil-code/dialogsum-test**
  - Training Set: 1,999 dialogues
  - Validation Set: 499 dialogues
  - Test Set: 499 dialogues
  - Features: id, dialogue, summary, topic

- **abisee/cnn_dailymail**
  - Training Set: 287,113 articles
  - Validation Set: 13,368 articles
  - Test Set: 11,490 articles
  - Features: article, highlights, id

### Key Steps

- **Model Upload**: Load the Microsoft-phi2 tokenizer and model.
- **Dataset Preparation**: Preprocess and tokenize the dataset.
- **Training**: Perform PEFT of the Microsoft-phi2 model.
- **Evaluation**: Compute ROUGE score for performance assessment.
- **Inference**: Generate and compare summaries on the `abisee/cnn_dailymail` dataset.

---

## Full-Fine Tuning Example Notebook for T5.1 and T5.2 Models

### Overview

This notebook illustrates full-fine tuning for T5.1 and T5.2 models. It includes dialogue summary fine-tuning and the use of pre-tuned models to streamline the tuning process.

### Datasets

- **BBC News Summary**
  - Contains 417 political news articles (2004-2005).
  - Folders: `Articles`, `Summaries`

- **DialogSum**
  - Contains 13,460 dialogues with 1,000 dialogues reserved for testing.
  - Folders: `dialogue/`, `summary/`, `topic/`

### Key Steps

1. **Model Upload**: Load T5 tokenizer and model.
2. **Dataset Preparation**: Preprocess and tokenize datasets.
3. **Training**: Full-fine tune T5 models.
4. **Evaluation**: Compute metrics such as ROUGE.
5. **Inference**: Generate and compare summaries using fine-tuned models.

---

## Full-Fine Tuning and PEFT Tuning Example Notebook for Google T5 Flan

### Overview

This notebook covers full-fine tuning and PEFT for the Google T5 model, utilizing both basic and complex Lora settings.

### Datasets

- **BBC News Summary**
  - Contains 417 political news articles (2004-2005).
  - Folders: `Articles`, `Summaries`

### Key Steps

1. **Model Upload**: Load Google T5 model and tokenizer.
2. **Dataset Preparation**: Preprocess and tokenize the dataset.
3. **Full-Fine Tuning**: Fine-tune Google T5 model.
4. **PEFT Tuning**:
   - **Basic Settings**: `r` and `lora_alpha` set to 32.
   - **Complex Settings**: `r` and `lora_alpha` set to 64.
5. **Evaluation**: Use ROUGE for model performance evaluation.
6. **Inference**: Generate summaries and compare model performance.

### Details on PEFT Tuning

- **Basic PEFT**: Tuning `r` and `lora_alpha` to 32.
- **Complex PEFT**: Extensive tuning with `r` and `lora_alpha` set to 64.

---

## PEFT Notebook for GPT-2

### Overview

This notebook illustrates the PEFT procedure for the GPT-2 model using the `neil-code/dialogsum-test` dataset. Performance is evaluated using ROUGE scores.

### Datasets

- **neil-code/dialogsum-test**
  - Training Set: 1,999 dialogues
  - Validation Set: 499 dialogues
  - Test Set: 499 dialogues
  - Features: id, dialogue, summary, topic

- **abisee/cnn_dailymail**
  - Training Set: 287,113 articles
  - Validation Set: 13,368 articles
  - Test Set: 11,490 articles
  - Features: article, highlights, id

### Key Steps

- **Model Upload**: Import GPT-2 model and tokenizer.
- **Dataset Preparation**: Tokenize and preprocess dataset.
- **Training**: Fine-tune GPT-2 model.
- **Evaluation**: Compute ROUGE score.
- **Inference**: Generate and compare summaries with refined models.

---

## Integration with FastAPI and Streamlit

### Setup

1. **FastAPI**
   - Develop a FastAPI application to serve the model through RESTful endpoints.
   - Endpoints include:
     - **GET /**: Returns a welcome message.
     - **POST /summarize/**: Accepts text input and returns a model-generated summary.

2. **Streamlit**
   - Build a Streamlit application to create an interactive web interface.
   - Features include:
     - **Text Input**: Input text for summarization.
     - **Summarize Button**: Generates summary and displays it in the interface.

### Running FastAPI

- Create and configure FastAPI to handle model inference requests.
- Start the FastAPI server to expose the model via HTTP.

### Running Streamlit

- Create a Streamlit app for user interaction.
- Launch the Streamlit server to view the web interface.

