# embeddig a string with fastembed
from fastembed.embedding import FlagEmbedding as Embedding
from typing import List
import numpy as np
def embed_str(input:str):
    """
    Function embedding the input to a 384 length numpy array,
    used for semantic search in Quadrant vector database.
    This is needed fro RAG (Retrieval Augmented Generation)
    :input: The short description of a context (String type)
    :return: numpy.Array.
    """
    embedding_model = Embedding(model_name='BAAI/bge-small-en-v1.5')
    return list(embedding_model.embed(input))[0]
