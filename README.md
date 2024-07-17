
# Zero-Shot Learning for Text Summarization: Improving Google FLAN's Performance through PEFT Fine-tuning for Adaptive Summary Generation and Classification

---

## PEFT Tuning Example Notebook for Microsoft-phi2
### Overview
This notebook demonstrates the process of parameter-efficient fine-tuning (PEFT) for the Microsoft-phi2 model using the neil-code/dialogsum-test dataset. The objective is to fine-tune the Microsoft-phi2 model for dialogue summarization tasks and evaluate its performance using the ROUGE score. Inference testing is performed on the abisee/cnn_dailymail dataset to assess zero-shot learning capabilities.

### Zeroshot Learning Inferences
Refer to the 2 Notebooks (finetune-phi-2-on-custom-dataset-v2.ipynb & finetune_phi_2_on_custom_dataset_V1.ipynb) for detailed explanations and examples of zero-shot inferences. This section explores how the Microsoft-phi2 model performs zero-shot tasks, providing insights into its capabilities without explicit training on the specific tasks or datasets used for inference.

### Datasets
#### neil-code/dialogsum-test
- Contains 2,997 dialogues for training, validation, and testing.
- Training set: 1,999 dialogues
- Validation set: 499 dialogues
- Test set: 499 dialogues
- Organized into four features: id, dialogue, summary, and topic.

#### abisee/cnn_dailymail
- Contains news articles and their highlights, used for inference testing
- Training set: 287,113 articles
- Validation set: 13,368 articles
- Test set: 11,490 articles
- Organized into three features: article, highlights, and id.

### Key Steps
- **Model Upload** : Load the Microsoft-phi2 tokenizer and model.
- **Dataset Preparation**: Preprocess and tokenize the neil-code/dialogsum-test dataset.
- **Training**: Perform parameter-efficient fine-tuning of the Microsoft-phi2 model with specified training parameters.
- **Evaluation**: Compute the ROUGE score to evaluate model performance.
- **Inference**: Generate summaries using both the original and fine-tuned models for comparison on the abisee/cnn_dailymail dataset.

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

---

## PEFT Notebook for GPT-2
### Overview
Using the neil-code/dialogsum-test dataset, the notebook 'gpt2-finetuning.ipynb' illustrates the parameter-efficient fine-tuning (PEFT) procedure for the GPT-2 model. The objective is to evaluate the GPT-2's performance using the ROUGE score. 

### Zeroshot Learning Inferences
For thorough explanations and examples of zero-shot inferences, consult the notebook 'gpt2-inference.ipynb'. The GPT-2 model's performance on zero-shot tasks is examined in this section, offering insights into its capabilities without the need for explicit training on the particular tasks or datasets used for inference.

### Datasets
#### neil-code/dialogsum-test
- Contains 2,997 dialogues for training, validation, and testing.
- Training set: 1,999 dialogues
- Validation set: 499 dialogues
- Test set: 499 dialogues
- Organized into four features: id, dialogue, summary, and topic.

#### abisee/cnn_dailymail
- Contains news articles and their highlights, used for inference testing
- Training set: 287,113 articles
- Validation set: 13,368 articles
- Test set: 11,490 articles
- Organized into three features: article, highlights, and id.

### Key Steps
- **Model Upload**: Import the model and tokenizer for GPT-2.
- **Dataset Preparation**: The neil-code/dialogsum-test dataset is tokenized and preprocessed.
- **Training**: Use the given training parameters to fine-tune the GPT-2 model in an efficient manner.
- **Evaluation**: To assess the performance of the model, compute the ROUGE score.
- **Inference**: Using the abisee/cnn_dailymail dataset (V 3.0.0), generate summaries utilizing both the original and refined models for comparison.

