
# Zero-Shot Learning for Text Summarization: Improving Google FLAN's Performance through PEFT Fine-tuning for Adaptive Summary Generation and Classification

---

## PEFT Tuning Example Notebook for Microsoft-phi2






---


## Full-Fine Tuning Example Notebook for T5.1 and T5.2 Models

### Overview
This notebook demonstrates the process of full-fine tuning for T5.1 and T5.2 models. Dialog Summary Fine-tuning (Second Process) was applied using the same notebook. The key distinction here is the inclusion of an already tuned model and its corresponding tokenizer to facilitate seamless continuation from previous tuning stages.

### Zeroshot Learning Inferences
Refer to the Inference Notebook for detailed explanations and examples of zero-shot inferences. This section explores how T5 models, specifically T5.1 and T5.2, perform zero-shot tasks, providing insights into their capabilities without explicit training on specific tasks or datasets.

### Datasets

#### BBC News Summary
- Contains 417 political news articles from BBC, spanning from 2004 to 2005.
- Organized into two main folders: `Articles` and `Summaries`.

#### DialogSum
- A large-scale dialogue summarization dataset comprising 13,460 dialogues, with an additional 1,000 dialogues reserved for testing purposes.
- Organized into three folders: `dialogue/`, `summary/`, and `topic/`.

### Key Steps
1. **Model Upload**: Loading T5 tokenizer and model.
2. **Dataset Preparation**: Preprocessing and tokenizing the BBC and DialogSum datasets.
3. **Training**: Full-fine tuning the T5 model with specified training parameters.
4. **Evaluation**: Computing metrics such as ROUGE to evaluate the model performance.
5. **Inference**: Generating summaries using both the original and fine-tuned models for comparison.

---

## Full-Fine Tuning and PEFT Tuning Example Notebook for Google T5 Flan

### Overview
This notebook demonstrates the process of Full-Fine tuning and Parameter-Efficient Fine-Tuning (PEFT) for the Google T5 model, using both basic and complex Lora settings.

### Zeroshot Learning Inferences
Refer to the Inference Notebook for detailed explanations and examples of zero-shot inferences.

### Datasets

#### BBC News Summary
- Contains 417 political news articles from BBC, spanning from 2004 to 2005.
- Organized into two main folders: `Articles` and `Summaries`.

### Key Steps
1. **Model Upload**: Loading the Google T5 model and tokenizer.
2. **Dataset Preparation**: Preprocessing and tokenizing the BBC dataset.
3. **Full-Fine Tuning**: Fine-tuning the Google T5 model with specified training parameters.
4. **PEFT Tuning**: 
   - **Basic Settings**: Tuning key hyperparameters such as `r` and `lora_alpha` to 32.
   - **Complex Settings**: More extensive tuning with `r` and `lora_alpha` set to 64.
5. **Evaluation**: Evaluating model performance using metrics like ROUGE.
6. **Inference**: Generating summaries to compare the performance of original and fine-tuned models.

### Details on PEFT Tuning
- **Basic PEFT**: Focused on two hyperparameters, `r` and `lora_alpha`, set to 32.
- **Complex PEFT**: Extensive tuning with `r` and `lora_alpha` set to 64.
- Both configurations targeted modules "q" and "v", employed a dropout rate of 0.05, and were tailored for sequence-to-sequence language modeling.

