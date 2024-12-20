import fitz  # PyMuPDF
import re
from data_ingestion.chonkie import semantic_chunks_by_chonkie

# PDF Parsing
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


# Text Preprocessing
def preprocess_text(text):
    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)  # Reduce multiple newlines to a single one

    # Remove page numbers (simple page number regex: change the pattern as needed)
    text = re.sub(r'\s*\d+\s*$', '', text)  # Remove page numbers at the end of each page

    # Refined footer pattern to handle spaces, line breaks, and subtle variations
    footer_pattern = r"Brevitaz Systems Pvt\. Ltd.*?http://brevitaz\.com"

    # Using re.sub to remove the footer, accounting for multiline and variations
    text = re.sub(footer_pattern, "", text, flags=re.DOTALL)

    # Strip extra white space and return
    return text.strip()


def create_chunks_from_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    text = preprocess_text(text)
    chunks = semantic_chunks_by_chonkie(text)
    # for idx, chunk in enumerate(chunks):
    #     print(f"Chunk {idx + 1}:\n{chunk}\n{'-' * 50}")
    return chunks

