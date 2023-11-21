from torch.utils.data import DataLoader
from index import Index
from sentence_encoder import SentenceEncoder
import pandas as pd

import nltk
import string
from nltk.tokenize.destructive import NLTKWordTokenizer
_treebank_word_tokenizer = NLTKWordTokenizer()


def read_file(file_name):
    texts = []
    with open(file_name, "r") as f:
        for l in f:
            texts.append(l.strip())
    return texts


def create_index():
    encoder = SentenceEncoder()
    texts = []

    for file_name in ["../resources/doc1.txt", "resources/doc2.txt"]:
        texts += tokenize(read_file(file_name))
    texts += process_csv()

    embeddings = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        embs = encoder.encode(texts[i: i + batch_size])
        embeddings += embs.tolist()
    
    index_ = Index(texts, embeddings)
    file_name = "../resources/index_checkpoint"
    print(f"Save checkpoint: {file_name}")
    index_.save(file_name)
    

def tokenize(texts):
    tokenizer = nltk.load(f"../resources/english.pickle")
    PUNCTUATION = set(string.punctuation)
    min_length = 10
    n_sent = 3

    texts = [tokenizer.tokenize(ref) for ref in texts]
    texts = [
        [token for sent in sentences for token in _treebank_word_tokenizer.tokenize(sent)] for sentences in texts
    ]

    texts = [
        [w.lower().translate(str.maketrans('', '', string.punctuation)) for w in ref if w not in PUNCTUATION]
        for ref in texts
    ]
    texts = [" ".join(t) for t in texts]
    texts = [t for t in texts if len(t) > min_length]
    new_texts = []
    for i in range(0, len(texts), n_sent):
        new_texts.append(" ".join(texts[i: i + n_sent]))
    return new_texts


def process_csv():
    dt = pd.read_csv("../resources/cards.csv")
    keys = ["Сервис", "Условия", "Тариф"]

    list_s = []
    for i in range(len(dt)):
        s = ""
        for k in keys:
            s += f"{k}: {dt.iloc[i][k]}"
        list_s.append(s)
    return list_s


create_index()

