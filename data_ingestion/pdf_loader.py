import fitz  # PyMuPDF
import re


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


# Split Text into Chunks
def split_text_into_chunks(text, max_chunk_length=500):
    chunks = []
    text = preprocess_text(text)
    sentences = text.split('\n\n')

    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
            current_chunk += (sentence + '\n')
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '\n'

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def create_chunks_from_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text, max_chunk_length=500)
    # for idx, chunk in enumerate(chunks):
    #     print(f"Chunk {idx + 1}:\n{chunk}\n{'-' * 50}")
    return chunks

