import math
import re


class BM25Retriever:

    def __init__(self, corpus: list[str], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []

        # Tokenize corpus
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)

        # Calculate term frequencies per document
        for doc in tokenized_corpus:
            frequencies = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)

            # Count document frequencies for IDF
            for word in set(doc):
                self.idf[word] = self.idf.get(word, 0) + 1

        # Calculate IDF using BM25 formula
        for word, freq in self.idf.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def get_scores(self, query: str) -> list[float]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        for i in range(self.corpus_size):
            doc_len = self.doc_lens[i]
            frequencies = self.doc_freqs[i]
            score = 0.0
            for token in query_tokens:
                if token in frequencies:
                    freq = frequencies[token]
                    idf_val = self.idf.get(token, 0.0)
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (
                        1 - self.b + self.b * doc_len / self.avg_doc_len
                    )
                    score += idf_val * numerator / denominator
            scores[i] = score
        return scores

    def retrieve(self, query: str, top_k=20) -> list[tuple[int, float]]:
        scores = self.get_scores(query)
        # Pair with index and sort
        indexed_scores = list(enumerate(scores))
        # Sort by score descending
        sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
