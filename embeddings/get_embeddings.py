from langchain_community.vectorstores import FAISS
import pickle

def load_faiss_index(index_path, metadata_path, embeddings):
    # Load the FAISS index
    faiss_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    # Load metadata
    with open(metadata_path, "rb") as f:
        faiss_db.docstore = pickle.load(f)
    return faiss_db
