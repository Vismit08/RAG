from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from data_ingestion.pdf_loader import create_chunks_from_pdf
from embeddings.create_embeddings import save_faiss_index
from embeddings.get_embeddings import load_faiss_index
from models.openai_gpt import get_openai_embedding_model
from config import INDEX_PATH, METADATA_PATH, DATA_PATH

def create_faiss_index_from_directory(pdf_dir, embedding_model, index_path, metadata_path):
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print("Loading FAISS index...")
        return load_faiss_index(index_path, metadata_path, embedding_model)

    print("Creating new FAISS index...")
    all_docs = []
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            print(f"Processing {pdf_file}...")
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text_chunks = create_chunks_from_pdf(pdf_path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            all_docs.extend(text_splitter.split_text("\n".join(text_chunks)))

    faiss_db = FAISS.from_texts(all_docs, embedding_model)
    save_faiss_index(faiss_db, index_path, metadata_path)
    return faiss_db


def main():
    embedding_model = get_openai_embedding_model()
    faiss_db = create_faiss_index_from_directory(DATA_PATH, embedding_model, INDEX_PATH, METADATA_PATH)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.9,
        max_tokens=256,
        top_p=0.9,

    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_db.as_retriever(),
    )

    while True:
        query = input("Please ask a question (or type 'exit' to quit): ")

        if query.lower() == 'exit':
            print("Exiting the program...")
            break

        response = qa_chain.invoke(query)
        print(f"Answer: {response.get('result')}")


if __name__ == '__main__':
    main()