import pandas as pd
from nltk import corpus
from bm25 import BM25
import os
from tabulate import tabulate

def extract_title(text: str) -> str:
    words = text.split(" ")
    i = 0
    while i < len(words) and words[i].upper() == words[i]:
        i += 1
    return " ".join(words[:i])


# Load Reuters corpus into a pandas dataframe
file_ids = corpus.reuters.fileids()
all_reuters_words = []
for file_id in file_ids:
    file_words = corpus.reuters.words(file_id)
    output = " ".join(file_words)
    all_reuters_words.append([file_id, output])
df_reuters = pd.DataFrame(all_reuters_words, columns=["ids", "text"])
df_reuters = df_reuters[:500]
df_reuters["title"] = df_reuters["text"].apply(extract_title)

bm25 = BM25(df_reuters, 1.25, 0.75, 1, rf_docs=5)

while True:
    os.system('clear')
    query = input("Enter a plain-text query: ")
    results = bm25.query(query, 45)
    print(tabulate(results, headers=["ids", "title", "score"]))
    input("Press Enter to continue...")