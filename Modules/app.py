import streamlit as st
import requests
import pandas as pd
from PIL import Image
import pytesseract
import io
import pdfplumber

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def extract_text_from_image(uploaded_file):
    """Extract text from an image using pytesseract."""
    try:
        image = Image.open(uploaded_file)
        image = image.convert('L')
        result = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        return result
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def main():
    st.title("Zero-Shot Text Summarizer")
    st.sidebar.title("Model, Prompt, and Input Type Selection")
    
    model_option = st.sidebar.selectbox("Select the Model", ["Google-Base", "Google-Peft-Lora", "T5-Double-Tuned", "Google-Fine-Tuned"], key='model_type')
    
    if model_option:
        prompt_option = st.sidebar.selectbox("Select the Prompt", ["Basic Prompt", "Classification Prompt", "Reasoning (COT) Prompt"], key='prompt')
        
        if prompt_option:
            input_type = st.sidebar.selectbox("Select Input Type", ["PDF", "Image", "CSV", "Plain Text"], key='input_type')
            
            st.title(f"Selected Model: {model_option}")
            st.subheader(f"Selected Prompt: {prompt_option}")
            
            if input_type == "PDF":
                st.subheader("PDF Input")
                uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
                if uploaded_pdf and st.button("Generate Summary"):
                    extracted_text = extract_text_from_pdf(uploaded_pdf)
                    st.text_area("Extracted Text from PDF", extracted_text)
                    data = {'input_text': extracted_text, 'model_type': model_option, 'prompt': prompt_option}
                    response = requests.post("http://127.0.0.1:8000/summarize", data=data)
                    if response.ok:
                        st.text_area("Output", response.json().get("summarized_text", ""), disabled=False)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

            elif input_type == "Image":
                st.subheader("Image Input")
                uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
                if uploaded_image and st.button("Generate Summary"):
                    extracted_text = extract_text_from_image(uploaded_image)
                    st.text_area("Extracted Text from Image", extracted_text)
                    data = {'input_text': extracted_text, 'model_type': model_option, 'prompt': prompt_option}
                    response = requests.post("http://127.0.0.1:8000/summarize", data=data)
                    if response.ok:
                        st.text_area("Output", response.json().get("summarized_text", ""), disabled=False)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

            elif input_type == "CSV":
                st.subheader("CSV Input")
                uploaded_csv = st.file_uploader("Upload CSV", type="csv")
                if uploaded_csv and st.button("Generate Summary"):
                    files = {'file': (uploaded_csv.name, uploaded_csv, 'text/csv')}
                    data = {'model_type': model_option, 'prompt': prompt_option}
                    response = requests.post("http://127.0.0.1:8000/upload-csv", files=files, data=data)
                    
                    if response.ok:
                        results = response.json()
                        st.write("Summarization completed. Check the results below.")
                        
                        st.subheader("Summarized Text")
                        if "data_frame" in results:
                            summaries_df = pd.DataFrame(results["data_frame"])
                            st.write(summaries_df)
                        else:
                            st.write("No summarized text available.")
                        
                        st.subheader("ROUGE Scores")
                        rouge_scores = results.get("rouge_scores", {})
                        if rouge_scores:
                            rouge_df = pd.DataFrame([rouge_scores])
                            st.dataframe(rouge_df)  
                        else:
                            st.write("No ROUGE scores available.")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

            elif input_type == "Plain Text":
                st.subheader("Plain Text Input")
                plain_text = st.text_area("Input Text")
                if plain_text and st.button("Generate Summary"):
                    data = {'input_text': plain_text, 'model_type': model_option, 'prompt': prompt_option}
                    response = requests.post("http://127.0.0.1:8000/summarize", data=data)
                    if response.ok:
                        st.text_area("Output", response.json().get("summarized_text", ""), disabled=False)
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
if __name__ == "__main__":
    main()
