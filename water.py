import gensim.downloader as api
from gensim.models import KeyedVectors

def cosine_similarity(model, word1, word2):
    return model.similarity(word1, word2)

# Pre-trained word embeddings model download
model = api.load("glove-wiki-gigaword-50")  # 50D GloVe embeddings

# Words for similarity check
word1 = "dirty"
word2 = "mango"

# Compute cosine similarity
similarity = cosine_similarity(model, word1, word2)
print(f"Cosine Similarity between {word1} and {word2}: {similarity:.4f}")

