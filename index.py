import pickle
import numpy as np

class Index:
    def __init__(self, sentences=[], embeddings=[]):
        self.list_of_str = sentences
        self.list_of_embs = embeddings
        self.array_of_embs = np.array(self.list_of_embs)

    def update_index(self, sentences=[], embeddings=[]):
        self.list_of_str += sentences
        self.list_of_embs += embeddings
        self.array_of_embs = np.array(self.list_of_embs)

    def find_top_k(self, embedding, top_k, min_sim):
        embs = self.array_of_embs / np.sqrt(np.sum(self.array_of_embs ** 2, axis=1)).reshape((-1, 1))
        embedding = embedding / np.sqrt(np.sum(embedding ** 2))
        similarities = embs.dot(embedding)
        inds = np.argpartition(similarities, -top_k)[-top_k:]
        sentences = [self.list_of_str[ind] for i, ind in enumerate(inds) if similarities[i] > min_sim]
        return sentences
    
    def save(self, file_name):
        state = {
            "sentences": self.list_of_str,
            "embeddings": self.list_of_embs
        }
        with open(f"{file_name}.pickle", "wb") as f:
            pickle.dump(state, f)
    
    def from_pretrained(self, file_name):
        with open(f"{file_name}.pickle", "rb") as f:
            state = pickle.load(f)

            self.list_of_str = state["sentences"]
            self.list_of_embs = state["embeddings"]
            self.array_of_embs = np.array(self.list_of_embs)
        return self
