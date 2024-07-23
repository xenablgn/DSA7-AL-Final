from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from io import StringIO
from model_utils import load_model_and_tokenizer
from evaluation import summarize_and_evaluate
from utils import BASIC_PROMT, CLASS_PROMT, COT_PROMT

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


app = FastAPI()

@app.post("/summarize")
async def summarize(
    input_text: str = Form(...),
    model_type: str = Form(...),
    prompt: str = Form(...)
):
    if not input_text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        print(f"Model: {model_type}")
        print(f"Text: {input_text[:100]}...")  # Print a truncated version of the text for debugging
        print(f"Prompt: {prompt}")
        
        prompt_template = ""
        if prompt == "Basic Prompt":
            prompt_template = BASIC_PROMT
        elif prompt == "Classification Prompt":
            prompt_template = CLASS_PROMT
        elif prompt == "Reasoning (COT) Prompt":
            prompt_template = COT_PROMT
        else:
            raise ValueError("Invalid prompt type")
        print(f"Prompt Variable: {prompt_template}")

        model, tokenizer = load_model_and_tokenizer(model_type)
        
        human_baseline_summaries = []
        model_output, rouge_scores = summarize_and_evaluate(
            model, tokenizer, [input_text], human_baseline_summaries, model_type, prompt_template
        )
        summarized_text = str(model_output.iloc[0, 2])

    except Exception as e:
        print(f"Error: {str(e)}")  # Print detailed error message
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
    
    return JSONResponse(content={"summarized_text": summarized_text})

@app.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    prompt: str = Form(...)
):
    try:
        prompt_templates = {
            "Basic Prompt": BASIC_PROMT,
            "Classification Prompt": CLASS_PROMT,
            "Reasoning (COT) Prompt": COT_PROMT
        }
        prompt_template = prompt_templates.get(prompt)
        print(f"Prompt Variable: {prompt_template}")
        if not prompt_template:
            raise ValueError("Invalid prompt type")

        df = pd.read_csv(file.file)
        if df.shape[1] < 1:
            raise ValueError("CSV file must contain at least one column")

        input_texts = df.iloc[:, 0].tolist()
        human_baseline_summaries = df.iloc[:, 1].tolist()
        input_texts = [text if pd.notna(text) else "" for text in input_texts]
        human_baseline_summaries = [text if pd.notna(text) else "" for text in human_baseline_summaries]

        model, tokenizer = load_model_and_tokenizer(model_type)
        model_output, rouge_scores = summarize_and_evaluate(
            model, tokenizer, input_texts, human_baseline_summaries, model_type, prompt_template
        )

        csv_buffer = StringIO()
        model_output.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return JSONResponse(
            content={
                "data_frame": model_output.to_dict(orient="records"),
                "rouge_scores": rouge_scores
            }
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Print detailed error message
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
