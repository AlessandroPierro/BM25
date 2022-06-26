import os
import time
import argparse
import pandas as pd
from nltk import corpus
from bm25 import BM25
from tabulate import tabulate


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ndocs', type=int, default=2500,
                        help='Number of documents to load from the corpus')
    parser.add_argument('--k1', type=float, default=1.2,
                        help='Coefficient in the BM25 formula')
    parser.add_argument('--b', type=float, default=0.75,
                        help='Coefficient in the BM25 formula')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Coefficient in the BM25+ formula')
    parser.add_argument('--rf_docs', type=int, default=15,
                        help='Number of documents to use for pseudo-relevance feedback')
    parser.add_argument('--rf_terms', type=int, default=20,
                        help='Number of terms to use for pseudo-relevance feedback')
    parser.add_argument('--nresults', type=int, default=25,
                        help='Number of documents to return')
    parser.add_argument('--expand', type=bool, default=False,
                        help='Expand the query with pseudo-relevance feedback')
    args = parser.parse_args()
    return args


def extract_title(text: str) -> str:
    i = 0
    while i < len(text) and text[i].upper() == text[i]:
        i += 1
    return text[:i]


if __name__ == "__main__":

    args = get_args()

    # Load the corpus
    file_ids = corpus.reuters.fileids()
    data = []
    for i in range(min(args.ndocs, len(file_ids))):
        words = corpus.reuters.words(file_ids[i])
        text = " ".join(words)
        title = extract_title(text)
        data.append([file_ids[i], title, text])
    df = pd.DataFrame(data, columns=['id', 'title', 'text'])

    # Create a BM25 index
    bm25 = BM25(df, args.k1, args.b, args.delta, args.rf_docs, args.rf_terms)

    # Perform queries
    while True:
        os.system('clear')
        query = input("Enter a plain-text query: ")
        start = time.time()
        results = bm25.query(query, args.nresults, expand=args.expand)
        end = time.time()
        print("\nQuery time: %.2f seconds\n" % (end - start))
        print(tabulate(results, headers=[
              "id", "title", "score"], showindex=False))
        if input("\nPress enter to continue or q to quit: ") == "q":
            break
