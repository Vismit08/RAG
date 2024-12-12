import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT= os.getenv("SYSTEM_PROMPT")
INDEX_PATH= os.getenv("INDEX_PATH")
METADATA_PATH= os.getenv("METADATA_PATH")
TIMESTAMP_PATH= os.getenv("TIMESTAMP_PATH")
DATA_PATH= os.getenv("DATA_PATH")
PERSIST_DIRECTORY= os.getenv("PERSIST_DIRECTORY")
COLLECTION_NAME= os.getenv("COLLECTION_NAME")