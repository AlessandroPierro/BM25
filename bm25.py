import re
import math
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from collections.abc import Set

STOP_WORDS = set(stopwords.words('english'))


class BM25(object):
    """
    Class to implement bm25 ranking algorithm, given a corpus of documents.
    """

    def __init__(self, corpus: pd.DataFrame, k: float, b: float, delta: float, rf_docs=None, rf_terms=25) -> None:
        """
        bm25 class initializer.

        params:
            corpus: Pandas DataFrame with a column "text"
            k (float): parameter controlling the impact of the term frequency

        """
        self.corpus = corpus
        self.k = k
        self.b = b
        self.delta = delta
        self.rf_docs = rf_docs
        self.rf_terms = rf_terms

        self.corpus["parsed_text"] = self.corpus["text"].apply(
            self.preprocess_text)
        self.corpus["word_count"] = self.corpus["parsed_text"].apply(
            lambda text: len(text.split(" ")))
        self.avg_word_count = self.corpus["word_count"].mean()

        self.vocabulary = BM25.build_vocabulary(corpus)
        self.tf = self.compute_tf()
        self.idf = BM25.compute_idf(self.tf, self.vocabulary, len(self.corpus))
        self.corpus.drop(columns=["parsed_text", "text"], inplace=True)

    @staticmethod
    def build_vocabulary(df: pd.DataFrame) -> Set[str]:
        """
        Given a Pandas DataFrame with a column "parsed_text", returns a set of all unique words in the corpus.

        params:
            df: Pandas DataFrame with a column "parsed_text"

        returns:
            vocabulary (set): set of all unique words in the corpus
        """
        vocabulary = set()
        df["parsed_text"].apply(
            lambda text: vocabulary.update(text.split(" ")))
        vocabulary.discard("")
        return vocabulary

    def compute_tf(self) -> pd.DataFrame:
        """
        Given a Pandas DataFrame with a column "parsed_text" and a set of unique words in the corpus,
        returns a DataFrame with a column for each unique word in the corpus, containing the number of times
        that word appears in the document.

        params:
            df: Pandas DataFrame with a column "parsed_text"
            vocabulary: set of unique words in the corpus

        returns:
            tf (pd.DataFrame): DataFrame with a column for each unique word in the corpus, containing the number of times
                               that word appears in the document.
        """
        term_frequency = pd.DataFrame(
            index=self.corpus["ids"], columns=self.vocabulary)
        for idx, row in self.corpus.iterrows():
            tmp = row["parsed_text"].split(" ")
            for term in set(tmp) - {""}:
                term_frequency.loc[row["ids"], term] = tmp.count(term)
        # for word in self.vocabulary:
        #    term_frequency[word] = self.corpus["parsed_text"].apply(lambda text: text.split(" ").count(word)).to_list()
        return term_frequency.fillna(0)

    @staticmethod
    def compute_idf(tf: pd.DataFrame, vocabulary: Set[str], n_docs: int) -> pd.DataFrame:
        """
        Given a Pandas DataFrame with a column for each unique word in the corpus, containing the number of times
        that word appears in the document, and a set of unique words in the corpus, returns a DataFrame with a row
        for each unique word in the corpus, containing the inverse document frequency of that word.

        $idf = log(N / (df + 1))$

        """
        vocabulary = list(vocabulary)
        idf = pd.DataFrame(index=vocabulary)
        idf["word"] = vocabulary
        idf["idf"] = idf["word"].apply(
            lambda word: math.log(n_docs / (tf[word].sum() + 1)))
        idf.drop(columns="word", inplace=True)
        return idf

    @staticmethod
    def preprocess_text(text: Set[str]) -> str:
        """
        Preprocess each string by:
        1. Lowercasing the string
        2. Removing punctuation and other non-alphanumeric characters
        3. Lemmatizing the string using WordNet's lemmatizer
        4. Removing stopwords (assumed to be English)
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9 ]", "", text)
        text = " ".join([WordNetLemmatizer().lemmatize(word)
                        for word in text.split(" ")])
        text = " ".join([word for word in text.split(" ")
                        if word not in STOP_WORDS])
        return text

    def score_document(self, corpus_row: pd.DataFrame, query_terms: Set[str]) -> float:
        score = 0
        for term in query_terms:
            if term in self.vocabulary:
                temp = self.score_term(corpus_row, term)
                score += temp
        return round(score, 4)

    def score_term(self, corpus_row: pd.DataFrame, term: str) -> float:
        """
        Given a corpus row and a term, returns the BM25 score for that term.

        params:
            corpus_row: corpus row
            term: term

        returns:
            score (float): BM25 score for that term
        """
        tf = self.tf.loc[corpus_row["ids"], term]
        term_idf = self.idf.loc[term, "idf"]
        return (self.k + 1) * tf * term_idf / ((self.k + tf) * (1 - self.b + self.b * corpus_row["word_count"] / self.avg_word_count))

    def query(self, query: str, max_results: int) -> pd.DataFrame:
        """

        """
        query = self.preprocess_text(query)
        query_terms = set(query.split(" "))
        results = self.corpus.copy().drop(columns="word_count")
        results["score"] = self.corpus.apply(
            lambda row: self.score_document(row, query_terms), axis=1)
        results = results[results["score"] > 0]
        results = results.sort_values(by="score", ascending=False)
        if self.rf_docs:
            rf_tf = self.tf.loc[results[:min(self.rf_docs, len(results))]["ids"], :].sum(
            ).transpose().sort_values(ascending=False)[:self.rf_terms]
            query_terms = query_terms.union(
                set(rf_tf.loc[rf_tf > 0].index.to_list()))
            results = self.corpus.copy().drop(columns="word_count")
            results["score"] = self.corpus.apply(
                lambda row: self.score_document(row, query_terms), axis=1)
            results = results[results["score"] > 0]
            results = results.sort_values(by="score", ascending=False)
        results = results[:min(max_results, len(results))]
        return results
