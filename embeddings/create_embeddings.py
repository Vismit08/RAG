import pickle

def save_faiss_index(faiss_db, index_path, metadata_path):
    # Save the FAISS index
    faiss_db.save_local(index_path)
    # Save metadata (e.g., document chunks)
    with open(metadata_path, "wb") as f:
        pickle.dump(faiss_db.docstore, f)
