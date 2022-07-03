from functools import lru_cache
import re
import math
import pandas as pd
from tqdm import tqdm
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

        tqdm.pandas(desc='{:<25}'.format('Preprocessing corpus'))
        self._corpus["parsed_text"] = self._corpus["text"].progress_apply(
            self._preprocess_text)
        tqdm.pandas(desc='{:<25}'.format('Computing word counts'))
        self._corpus["word_count"] = self._corpus["parsed_text"].progress_apply(
            lambda text: len(text.split(" ")))
        self._avg_word_count = self._corpus["word_count"].mean()

        tqdm.pandas(desc='{:<25}'.format('Building vocabulary'))
        self._vocabulary = self._build_vocabulary()
        self._tf = self._compute_tf()
        tqdm.pandas(desc='{:<25}'.format('Computing idf'))
        self._idf = self._compute_idf()
        self._corpus.drop(columns=["parsed_text", "text"], inplace=True)

    def _build_vocabulary(self) -> Set[str]:
        """
        Returns a set of all unique words in the corpus, based on the 'parsed_text' column of self.corpus.

        returns:
            vocabulary (set): set of all unique words in the corpus
        """
        vocabulary = set()
        self._corpus["parsed_text"].progress_apply(
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
        for _, row in tqdm(self._corpus.iterrows(), desc='{:<25}'.format("Computing term frequency"),
                           total=len(self._corpus)):
            tmp = row["parsed_text"].split(" ")
            for term in set(tmp) - {""}:
                term_frequency.loc[row["id"], term] = tmp.count(term)
        term_frequency.fillna(0, inplace=True)
        return term_frequency

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
        idf["idf"] = idf["word"].progress_apply(
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

        params:
            text (str): string to preprocess
        returns:
            text (str): preprocessed string
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
        return ((self.k1 + 1) * tf + self.delta) * term_idf / (
            (self.k1 + tf) * (1 - self.b + self.b * corpus_row["word_count"] / self._avg_word_count))

    def _compute_offer_weights(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the offer weights for each document in the results DataFrame to power the pseudo-relevance feedback.

        params:
            results (pd.DataFrame): DataFrame with a column "id" and a column "score" with the initial results.
        returns:
            offer_weights (pd.Series): Series with the offer weights for each term.
        """
        offer_weights = self._tf.loc[results['id'], :].astype(bool).sum(axis=0)
        for idx, value in offer_weights.iteritems():
            offer_weights[idx] = value * self._idf.loc[idx, "idf"]
        offer_weights = offer_weights.sort_values(ascending=False)
        return offer_weights

    @lru_cache
    def query(self, query: str, max_results: int, expand=False) -> pd.DataFrame:
        """
        Given a query string, return a dataframe with the top max_results results.

        Implements a BM25+ variant, where the pseudo-relevance feedback is implemented
        using the top rf_docs documents and the top rf_terms terms, based on the tf scores.

        params:
            query (str): query string
            max_results (int): maximum number of results to return
            expand (bool): if True, return the expanded results, otherwise return the original results
        returns:
            results (pd.DataFrame): dataframe with the top max_results results
        """
        query = self._preprocess_text(query)
        query_terms = set(query.split(" "))
        results = self._corpus.copy().drop(columns="word_count")
        results["score"] = self._corpus.apply(
            lambda row: self._score_document(row, query_terms), axis=1)
        results = results[results["score"] > 0]
        results = results.sort_values(by="score", ascending=False)
        if expand:
            results = results[:min(max_results, len(results))]
            offer_weights = self._compute_offer_weights(results)
            additional_terms = set(offer_weights.index[:self.rf_terms].to_list())
            query_terms = query_terms.union(additional_terms)
            new_query = " ".join(query_terms)
            print(f"Expanded query: {new_query}")
            return self.query(new_query, max_results, expand=False)
        results = results[:min(max_results, len(results))]
        return results
