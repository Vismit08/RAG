from config import OPENAI_API_KEY
import openai
from langchain.schema import AIMessage
from langchain_openai import OpenAIEmbeddings

openai.api_key = OPENAI_API_KEY


def get_openai_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-ada-002")
