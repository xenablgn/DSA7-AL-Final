import streamlit as st
import requests
import pandas as pd
from PIL import Image
import pytesseract
import pdfplumber

# Tesseract OCR path (update as necessary)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def extract_text_from_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        image = image.convert('L')
        result = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        return result
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def main():
    # Add custom CSS
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
            padding: 20px;
        }
        .stApp {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stTextArea>textarea {
            border: 1px solid #ced4da;
            border-radius: 5px;
            padding: 10px;
        }
        .stTextArea>textarea:focus {
            border-color: #80bdff;
            box-shadow: 0 0 0 0.2rem rgba(38, 143, 255, 0.25);
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Zero-Shot Text Summarizer")
    
    st.sidebar.title("Generation Options")
    
    model_option = st.sidebar.selectbox("Select the Model", ["Google-Base", "Google-Peft-Lora", "T5-Double-Tuned", "Google-Fine-Tuned"], key='model_type')
    
    if model_option:
        prompt_option = st.sidebar.selectbox("Select the Prompt", ["Basic Prompt", "Classification Prompt", "Reasoning (COT) Prompt"], key='prompt')
        
        if prompt_option:
            input_type = st.sidebar.selectbox("Select Input Type", ["PDF", "Image", "CSV", "Plain Text"], key='input_type')
            
            st.subheader(f"Selected Model: {model_option}")
            st.subheader(f"Selected Prompt: {prompt_option}")
            
            if input_type == "PDF":
                st.subheader("Upload PDF")
                uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
                if uploaded_pdf and st.button("Generate Summary"):
                    extracted_text = extract_text_from_pdf(uploaded_pdf)
                    st.text_area("Extracted Text from PDF", extracted_text, height=300)
                    data = {'input_text': extracted_text, 'model_type': model_option, 'prompt': prompt_option}
                    try:
                        response = requests.post("http://127.0.0.1:8000/summarize", data=data)
                        if response.ok:
                            summarized_text = response.json().get("summarized_text", "")
                            st.text_area("Summary", summarized_text, height=300, disabled=False)
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

            elif input_type == "Image":
                st.subheader("Upload Image")
                uploaded_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"], key="image_uploader")
                if uploaded_image and st.button("Generate Summary"):
                    extracted_text = extract_text_from_image(uploaded_image)
                    st.text_area("Extracted Text from Image", extracted_text, height=300)
                    data = {'input_text': extracted_text, 'model_type': model_option, 'prompt': prompt_option}
                    try:
                        response = requests.post("http://127.0.0.1:8000/summarize", data=data)
                        if response.ok:
                            summarized_text = response.json().get("summarized_text", "")
                            st.text_area("Summary", summarized_text, height=300, disabled=False)
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

            elif input_type == "CSV":
                st.subheader("Upload CSV")
                uploaded_csv = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")
                if uploaded_csv and st.button("Generate Summary"):
                    files = {'file': (uploaded_csv.name, uploaded_csv, 'text/csv')}
                    data = {'model_type': model_option, 'prompt': prompt_option}
                    try:
                        response = requests.post("http://127.0.0.1:8000/upload-csv", files=files, data=data)
                        if response.ok:
                            results = response.json()
                            st.write("Summarization completed. Check the results below.")
                            
                            st.subheader("Summarized Text")
                            if "data_frame" in results:
                                summaries_df = pd.DataFrame(results["data_frame"])
                                st.write(summaries_df)
                                csv = summaries_df.to_csv(index=False).encode('utf-8')
                                st.download_button(label="Download CSV", data=csv, file_name="summarized_text.csv", mime='text/csv')
                            else:
                                st.write("No summarized text available.")
                            
                            st.subheader("ROUGE Scores")
                            
                            rouge_descriptions = {
                                "rouge1": "Unigram (single word) overlap between the reference and the candidate text.",
                                "rouge2": "Bigram (two consecutive words) overlap between the reference and the candidate text.",
                                "rougeL": "Longest common subsequence between the reference and the candidate text.",
                                "rougeLsum": "Longest common subsequence (summary-based) between the reference and the candidate text."
                            }
                            
                            rouge_scores = results.get("rouge_scores", {})
                            if rouge_scores:
                                rouge_df = pd.DataFrame.from_dict(rouge_scores, orient='index').reset_index()
                                rouge_df.columns = ['ROUGE Score', 'Value']
                                rouge_df['Description'] = rouge_df['ROUGE Score'].map(rouge_descriptions)
                                st.dataframe(rouge_df)
                                csv = rouge_df.to_csv(index=False).encode('utf-8')
                                st.download_button(label="Download ROUGE Scores CSV", data=csv, file_name="rouge_scores.csv", mime='text/csv')
                            else:
                                st.write("No ROUGE scores available.")
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

            elif input_type == "Plain Text":
                st.subheader("Enter Plain Text")
                plain_text = st.text_area("Input Text", height=300)
                if plain_text and st.button("Generate Summary"):
                    data = {'input_text': plain_text, 'model_type': model_option, 'prompt': prompt_option}
                    try:
                        response = requests.post("http://127.0.0.1:8000/summarize", data=data)
                        if response.ok:
                            summarized_text = response.json().get("summarized_text", "")
                            st.text_area("Summary", summarized_text, height=300, disabled=False)
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

if __name__ == "__main__":
    main()
