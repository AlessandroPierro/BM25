import os
import time
import argparse
import pandas as pd
from pickle import load, dump
from nltk import corpus
from bm25 import BM25
from tabulate import tabulate

pd.options.plotting.backend = "plotly"


def get_args():
    parser = argparse.ArgumentParser(description='Information Retrieval system based on BM25.\n'
                                     'This program is a demo of the BM25 algorithm.')
    parser.add_argument('--ndocs', type=int, default=2500,
                        help='Maximum number of documents to load from the corpus')
    parser.add_argument('--k1', type=float, default=1.2,
                        help='Coefficient in the BM25 formula')
    parser.add_argument('--b', type=float, default=0.75,
                        help='Coefficient in the BM25 formula')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Coefficient in the BM25+ formula')
    parser.add_argument('--rf_docs', type=int, default=15,
                        help='Number of documents to use for pseudo-relevance feedback')
    parser.add_argument('--rf_terms', type=int, default=10,
                        help='Number of terms to use for pseudo-relevance feedback')
    parser.add_argument('--nresults', type=int, default=30,
                        help='Number of documents to return')
    parser.add_argument('--expand', type=bool, default=False,
                        help='Expand the query with pseudo-relevance feedback')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to a pickle file containing a previously-saved BM25 object')
    parser.add_argument('--dump', type=str, default=None,
                        help='Path to a pickle file to save the BM25 object')
    parser.add_argument('--plot', type=bool, default=False,
                        help='Plot the score distribution')
    args = parser.parse_args()
    return args


def extract_title(text: str) -> str:
    i = 0
    while i < len(text) and text[i].upper() == text[i]:
        i += 1
    i = min(i, 40)
    return text[:i]


if __name__ == "__main__":

    args = get_args()

    # Load the corpus
    if args.load is None:
        file_ids = corpus.reuters.fileids()
        data = []
        for i in range(min(args.ndocs, len(file_ids))):
            words = corpus.reuters.words(file_ids[i])
            text = " ".join(words)
            title = extract_title(text)
            data.append([file_ids[i], title, text])
        df = pd.DataFrame(data, columns=['id', 'title', 'text'])
        bm25 = BM25(df, args.k1, args.b, args.delta,
                    args.rf_docs, args.rf_terms)
        if args.dump is not None:
            with open(args.dump, 'wb') as f:
                dump(bm25, f)
    else:
        with open(args.load, 'rb') as f:
            bm25 = load(f)

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
        if args.plot:
            results.drop(['id', 'title'], axis=1, inplace=True)
            results.index = list(range(len(results)))
            fig = results.plot.bar()
            fig.update_layout(legend_title_text="Legend")
            fig.update_xaxes(title_text="Result Index")
            fig.update_yaxes(title_text="Score")
            fig.update_layout(title_text=f"Query Results for '{query}'")
            fig.show()
        if input("\nPress enter to continue or q to quit: ") == "q":
            break
