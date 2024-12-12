from chonkie import SemanticChunker


def semantic_chunks_by_chonkie(text):
    # Basic initialization with default parameters
    chunker = SemanticChunker(
        embedding_model="text-embedding-ada-002",  # Default model
        threshold=0.5,  # Similarity threshold (0-1)
        chunk_size=512,  # Maximum tokens per chunk
    )

    return chunker.chunk(text)
