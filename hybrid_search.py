import os
import re 
import pickle
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def _simple_tokenize(self, text: str):
    # Tokenizer for BM25 to ensure case-insensitive matching
    return re.findall(r"[a-z0-9]+", text.lower())

def _bm25_index_path(self):         #Generates the file path for saving/loading the BM25 index.
    return os.path.join(self.store_dir, "bm25_index.pkl")

def _load_bm25_index(self):
    """  Load BM25 index from disk if it exists. """
    p = self._bm25_index_path()
    if os.path.exists(p):
        with open(p, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._bm25_ids = data["ids"]
        self._bm25_raws = data["raws"]
        return True
    return False


def _save_bm25_index(self):
    if getattr(self, "_bm25", None) is None:
        return
    data = {"bm25": self._bm25, "ids": self._bm25_ids, "raws": self._bm25_raws}
    with open(self._bm25_index_path(), "wb") as f:
        pickle.dump(data, f)

def _build_bm25_index(self, ids, summaries, titles=None, headings=None):
    """
    Build BM25 over summary text, lightly enriching with title/heading if present.
    """
    corpus = []
    for i, summ in enumerate(summaries):
        pieces = [summ or ""]
        if titles:   pieces.append(titles[i] or "")
        if headings: pieces.append(headings[i] or "")
        corpus.append(" \n ".join(pieces))

    tokenized = [self._simple_tokenize(t) for t in corpus]
    self._bm25 = BM25Okapi(tokenized)
    self._bm25_ids = ids
    self._bm25_raws = corpus
    self._save_bm25_index()

def _bm25_search(self, query: str, top_k: int = 30):
    """
    Return [(doc_id, score, summary_text), ...] sorted by BM25 score desc.
    """
    if getattr(self, "_bm25", None) is None:
        # If BM25 index is not loaded, try to load it
        if not self._load_bm25_index():
            return []  # No index available   

    # Tokenize the query
    tokens = self._simple_tokenize(query)
    scores = self._bm25.get_scores(tokens)
    order = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in order:
        results.append((self._bm25_ids[idx], float(scores[idx]), self._bm25_raws[idx]))
    # Sort results by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    # Return only the top_k results
    results = results[:top_k]
    return results

def _hybrid_merge(self, emb_ids, emb_scores, bm25_ids, bm25_scores, alpha: float = 0.55, top_k: int = 10):
    """
    emb_scores, bm25_scores are dicts {id: score}; alpha weights embeddings.
    Returns list of ids sorted by hybrid score desc.
    """
    # Union of ids
    all_ids = sorted(set(emb_ids) | set(bm25_ids))
    if not all_ids:
        return []   # No results to merge
    # Create score arrays
    e = np.array([emb_scores.get(i, 0.0) for i in all_ids]).reshape(-1, 1)
    b = np.array([bm25_scores.get(i, 0.0) for i in all_ids]).reshape(-1, 1)

    # Normalize separately to [0,1] (robust with constant arrays)
    if e.size == 0 or b.size == 0:
        return []   # No scores to merge
    
    scaler_e = MinMaxScaler()
    scaler_b = MinMaxScaler()
    # Handle case where all scores are identical (avoid division by zero in MinMaxScaler)
    if np.all(e == e[0]):
        e_norm = e
    else:
        e_norm = scaler_e.fit_transform(e)
    if np.all(b == b[0]):
        b_norm = b
    else:
        b_norm = scaler_b.fit_transform(b)
    # Calculate hybrid score
    hybrid = alpha * e_norm.flatten() + (1 - alpha) * b_norm.flatten()
    e_norm = scaler_e.fit_transform(e) if (e.max() > e.min()) else e
    b_norm = scaler_b.fit_transform(b) if (b.max() > b.min()) else b

    # Sort by hybrid score and return top_k results
    hybrid = alpha * e_norm.flatten() + (1 - alpha) * b_norm.flatten()
    order = np.argsort(hybrid)[::-1][:top_k]
    # Return list of (id, hybrid_score, e_norm_score, b_norm_score)    
    return [(all_ids[i], float(hybrid[i]), float(e_norm[i]), float(b_norm[i])) for i in order]