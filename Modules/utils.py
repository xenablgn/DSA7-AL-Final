BASIC_PROMT = "Summarize the following news:\n\n{input}\n\nSummary: "
CLASS_PROMT = "Classify the following conversation into  topics such as 'business', 'sports', 'tech','politics' or 'entertainment'.\n\nConversation:\n\n{input}\n\nClassification: "
COT_PROMT = "Step 1: Identify the main theme of this passage:\n\n{input}\n\nBefore generating questions, identify the main themes in the text.\n\nStep 2: Summarise the theme\n\nSummary: "

MODELS= ['Google-Base', 'Google-Peft-Lora', 'T5-Double-Tuned', 'Google-Fine-Tuned','Microsoft-V1', 'Microsoft-V2']

MAX_SEQUENCE_LENGTHS = {
    'Google-Base': 512,  # Maximum length for T5 models
    'GOOGLE-PEFT-LORA': 512,  # Maximum length for T5 models with PEFT
    'T5-Double-Tuned': 512,  # Maximum length for T5 models
    'Google-Fine-Tuned': 512,  # Maximum length for T5 models
}


