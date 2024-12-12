from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema.document import Document
from data_ingestion.pdf_loader import create_chunks_from_pdf
from models.openai_gpt import get_openai_embedding_model
from embeddings.chroma_db import build_chroma_db
from config import DATA_PATH,PERSIST_DIRECTORY, COLLECTION_NAME
from langchain_community.vectorstores.chroma import Chroma

import os

def get_or_create_chroma_db(pdf_dir, embedding_model, persist_directory, collection_name):
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading existing ChromaDB...")
        return Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

    print("Creating new ChromaDB...")
    all_docs = []
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            print(f"Processing {pdf_file}...")
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text_chunks = create_chunks_from_pdf(pdf_path)
            text_chunks = [str(chunk) for chunk in text_chunks]

            all_docs.extend([Document(page_content=chunk) for chunk in text_chunks])

    chroma_db = build_chroma_db(
        collection_name=collection_name,
        embedding=embedding_model,
        chroma_server_ssl_enabled=False,
        index_directory=persist_directory,
        inputs=all_docs
    )

    return chroma_db

def main():
    embedding_model = get_openai_embedding_model()
    persist_directory = PERSIST_DIRECTORY  # Directory to persist ChromaDB
    collection_name = COLLECTION_NAME

    chroma_db = get_or_create_chroma_db(DATA_PATH, embedding_model, persist_directory, collection_name)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.9,
        max_tokens=256,
        top_p=0.9,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=chroma_db.as_retriever(),
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