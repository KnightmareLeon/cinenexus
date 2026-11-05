import numpy as np

def encode_texts(texts, model):
    return np.asarray(model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))

def encode_query(query, model):
    return encode_texts([query], model)