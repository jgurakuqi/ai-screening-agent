"""FAQ retrieval via semantic similarity using sentence-transformers.

Embeds FAQ entries at startup and performs cosine-similarity search at query time
to surface relevant answers during the screening conversation.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.logging_config import logger

# Module-level state — initialized once at startup
_model: SentenceTransformer | None = None
_embeddings: np.ndarray | None = None  # shape (N, 384), normalized
_faq_index: list[tuple[dict, str]] = []   # maps embedding row -> (faq_entry, lang)

FAQ_SIMILARITY_THRESHOLD = float(os.getenv("FAQ_SIMILARITY_THRESHOLD", "0.65"))
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def initialize(faq_path: str | None = None) -> None:
    """Load the sentence-transformer model and pre-compute FAQ embeddings.

    Must be called once at application startup before any calls to
    :func:`search`.

    Args:
        faq_path: Path to the FAQ JSON file. Defaults to
            ``data/faq.json`` relative to the project root.
    """
    global _model, _embeddings, _faq_index

    if faq_path is None:
        faq_path = str(Path(__file__).resolve().parent.parent / "data" / "faq.json")

    logger.info("Loading FAQ embedding model: {}", MODEL_NAME)
    faq_device = os.getenv("FAQ_DEVICE", "cpu")
    _model = SentenceTransformer(MODEL_NAME, device=faq_device)

    with open(faq_path, "r", encoding="utf-8") as f:
        faq_entries = json.load(f)

    logger.info("Loaded {} FAQ entries from {}", len(faq_entries), faq_path)

    texts: list[str] = []
    _faq_index = []
    for entry in faq_entries:
        texts.append(entry["question_es"])
        _faq_index.append((entry, "es"))
        texts.append(entry["question_en"])
        _faq_index.append((entry, "en"))

    _embeddings = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    logger.info("FAQ embeddings computed: {} vectors of dim {}", _embeddings.shape[0], _embeddings.shape[1])


def search(query: str, language: str = "es", threshold: float | None = None) -> dict | None:
    """Search FAQs for the best semantic match to the query.

    Args:
        query: The user's message text to match against FAQ entries.
        language: Language code (``"es"`` or ``"en"``) — determines which
            answer translation is returned.
        threshold: Minimum cosine-similarity score to accept a match.
            Defaults to ``FAQ_SIMILARITY_THRESHOLD`` from env.

    Returns:
        A dict with keys ``answer``, ``question``, ``score``, ``category``,
        and ``id`` if a match is found above the threshold, or ``None``
        otherwise.
    """
    if _model is None or _embeddings is None:
        logger.warning("FAQ not initialized — skipping search")
        return None

    if threshold is None:
        threshold = FAQ_SIMILARITY_THRESHOLD

    # Split query into sentences so non-question text doesn't dilute the embedding
    sentences = [s.strip() for s in re.split(r'[.!?\n]', query) if s.strip()]
    if not sentences:
        sentences = [query]

    query_emb = _model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    # query_emb shape: (num_sentences, dim); _embeddings shape: (num_faqs, dim)
    all_scores = np.dot(_embeddings, query_emb.T)  # (num_faqs, num_sentences)
    # Find the global maximum across all sentences and FAQ entries
    flat_idx = int(np.argmax(all_scores))
    best_idx = flat_idx // len(sentences)
    best_score = float(all_scores.flat[flat_idx])

    if best_score < threshold:
        logger.debug("FAQ search: best score {:.3f} below threshold {:.2f}", best_score, threshold)
        return None

    entry, matched_lang = _faq_index[best_idx]

    # Return answer in the candidate's language
    answer_key = f"answer_{language}"
    answer = entry.get(answer_key, entry.get("answer_en", ""))

    question_key = f"question_{language}"
    question = entry.get(question_key, entry.get("question_en", ""))

    logger.info(
        "FAQ match: id={} score={:.3f} category={} matched_lang={}",
        entry.get("id", "?"), best_score, entry.get("category", "?"), matched_lang,
    )

    return {
        "answer": answer,
        "question": question,
        "score": best_score,
        "category": entry.get("category", ""),
        "id": entry.get("id", ""),
    }
