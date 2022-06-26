from functools import lru_cache
import re
import math
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from collections.abc import Set

STOP_WORDS = set(stopwords.words('english'))


class BM25(object):
    """
    Class to implement BM25 information retrieval algorithm.
    """

    def __init__(self, corpus: pd.DataFrame, k1: float, b: float, delta: float, rf_docs: int, rf_terms: int) -> None:
        """
        BM25 class initializer.

        params:
            corpus: Pandas DataFrame with a column "text"
            k1 (float): parameter controlling the impact of the term frequency
            b (float): parameter controlling the impact of the document length
            delta (float): normalizing parameter introduced in the BM25+ variant
            rf_docs (int): number of documents to use for pseudo-relevance feedback
            rf_terms (int): number of terms to use for pseudo-relevance feedback
        """
        self._corpus = corpus
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.rf_docs = rf_docs
        self.rf_terms = rf_terms

        self._corpus["parsed_text"] = self._corpus["text"].apply(
            self._preprocess_text)
        self._corpus["word_count"] = self._corpus["parsed_text"].apply(
            lambda text: len(text.split(" ")))
        self._avg_word_count = self._corpus["word_count"].mean()

        self._vocabulary = self._build_vocabulary()
        self._tf = self._compute_tf()
        self._idf = self._compute_idf()
        self._corpus.drop(columns=["parsed_text", "text"], inplace=True)

    def _build_vocabulary(self) -> Set[str]:
        """
        Returns a set of all unique words in the corpus, based on the 'parsed_text' column of self.corpus.

        returns:
            vocabulary (set): set of all unique words in the corpus
        """
        vocabulary = set()
        self._corpus["parsed_text"].apply(
            lambda text: vocabulary.update(text.split(" ")))
        vocabulary.discard("")
        return vocabulary

    def _compute_tf(self) -> pd.DataFrame:
        """
        Create a dataframe with the term frequency of each term in the vocabulary.

        params:
            df: Pandas DataFrame with a column "parsed_text"
            vocabulary: set of unique words in the corpus

        returns:
            tf (pd.DataFrame): DataFrame with a column for each unique word in the corpus, containing the number of times
                               that word appears in the document.
        """
        term_frequency = pd.DataFrame(
            index=self._corpus["id"], columns=self._vocabulary)
        for _, row in self._corpus.iterrows():
            tmp = row["parsed_text"].split(" ")
            for term in set(tmp) - {""}:
                term_frequency.loc[row["id"], term] = tmp.count(term)
        return term_frequency.fillna(0)

    def _compute_idf(self) -> pd.DataFrame:
        """
        Compute the inverse document frequency of each term in the vocabulary,
        using the formula idf = log(N / df) + 1.

        returns:
            idf (pd.DataFrame): DataFrame with a column for each unique word in the corpus, containing the inverse document
                                    frequency of that word.
        """
        vocabulary = list(self._vocabulary)
        idf = pd.DataFrame(index=vocabulary)
        idf["word"] = vocabulary
        idf["idf"] = idf["word"].apply(
            lambda word: math.log(len(self._corpus) / (self._tf[word].sum() + 1)))
        idf.drop(columns="word", inplace=True)
        return idf

    @staticmethod
    def _preprocess_text(text: str) -> str:
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

    def _score_document(self, corpus_row: pd.DataFrame, query_terms: Set[str]) -> float:
        """
        Compute the BM25 score for a document, given a set of query terms.
        """
        score = 0
        for term in query_terms:
            if term in self._vocabulary:
                temp = self._score_term(corpus_row, term)
                score += temp
        return round(score, 4)

    def _score_term(self, corpus_row: pd.DataFrame, term: str) -> float:
        """
        Given a corpus row and a term, returns the BM25 score for that term.
        """
        tf = self._tf.loc[corpus_row["id"], term]
        term_idf = self._idf.loc[term, "idf"]
        return ((self.k1 + 1) * tf + self.delta) * term_idf / ((self.k1 + tf) * (1 - self.b + self.b * corpus_row["word_count"] / self._avg_word_count))

    @lru_cache
    def query(self, query: str, max_results: int, expand=False) -> pd.DataFrame:
        """

        """
        query = self._preprocess_text(query)
        query_terms = set(query.split(" "))
        results = self._corpus.copy().drop(columns="word_count")
        results["score"] = self._corpus.apply(
            lambda row: self._score_document(row, query_terms), axis=1)
        results = results[results["score"] > 0]
        results = results.sort_values(by="score", ascending=False)
        if expand:
            rf_tf = self._tf.loc[results[:min(self.rf_docs, len(results))]["id"], :].sum(
            ).transpose().sort_values(ascending=False)[:self.rf_terms]
            query_terms = query_terms.union(
                set(rf_tf.loc[rf_tf > 0].index.to_list()))
            return self.query(" ".join(query_terms), max_results, expand=False)
        results = results[:min(max_results, len(results))]
        return results
