"""Unit tests for FAQ knowledge base (embedding search + question detection)."""
import sys
import os
import json

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend import faq


# --- Initialization ---

def test_initialize():
    """FAQ loads without error and embeddings have correct shape."""
    faq.initialize()
    assert faq._model is not None
    assert faq._embeddings is not None
    # We have N FAQ entries, each with ES + EN question = 2N embeddings
    faq_path = os.path.join(os.path.dirname(__file__), "..", "data", "faq.json")
    with open(faq_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    expected_rows = len(entries) * 2
    assert faq._embeddings.shape[0] == expected_rows
    assert faq._embeddings.shape[1] == 384  # MiniLM-L12-v2 dimension


# --- Search (requires initialization) ---

def _ensure_initialized():
    if faq._model is None:
        faq.initialize()


def test_search_spanish_salary():
    _ensure_initialized()
    result = faq.search("¿Cuánto se paga por este trabajo?", language="es")
    assert result is not None
    assert result["category"] == "compensation"
    assert result["score"] >= 0.65


def test_search_english_salary():
    _ensure_initialized()
    result = faq.search("How much does the job pay?", language="en")
    assert result is not None
    assert result["category"] == "compensation"
    assert "answer" in result


def test_search_vehicle_question():
    _ensure_initialized()
    result = faq.search("¿Necesito mi propio coche?", language="es")
    assert result is not None
    assert result["category"] == "vehicle"


def test_search_training_question():
    _ensure_initialized()
    result = faq.search("Is there training? Is it paid?", language="en")
    assert result is not None
    assert result["category"] == "training"


def test_search_returns_correct_language():
    _ensure_initialized()
    result_es = faq.search("What is the pay?", language="es")
    result_en = faq.search("What is the pay?", language="en")
    assert result_es is not None and result_en is not None
    # Spanish answer should be in Spanish
    assert "salario" in result_es["answer"].lower() or "pago" in result_es["answer"].lower() or "repartidores" in result_es["answer"].lower()


def test_search_below_threshold():
    _ensure_initialized()
    result = faq.search("Me llamo Juan García López", language="es")
    assert result is None


def test_search_irrelevant_statement():
    _ensure_initialized()
    result = faq.search("Sí, tengo carnet de conducir desde hace 5 años", language="es")
    # This is a screening answer, not a question — should not match strongly
    # (it might weakly match vehicle/license FAQs, but threshold should filter it)
    if result is not None:
        assert result["score"] < 0.80  # shouldn't be a strong match


def test_search_custom_threshold():
    _ensure_initialized()
    # Very high threshold should return no results
    result = faq.search("¿Cuánto pagan?", language="es", threshold=0.99)
    assert result is None


def test_search_company_question():
    _ensure_initialized()
    result = faq.search("¿Qué es Grupo Sazón?", language="es")
    assert result is not None
    assert result["category"] == "company"


def test_search_process_question():
    _ensure_initialized()
    result = faq.search("What happens after the screening?", language="en")
    assert result is not None
    assert result["category"] == "application_process"


def test_search_question_embedded_in_greeting():
    """Question mixed with a greeting/answer — the original bug."""
    _ensure_initialized()
    result = faq.search("Hi, my name is Mauro. How much is the pay", language="en")
    assert result is not None
    assert result["category"] == "compensation"


def test_search_informal_no_question_mark():
    """Implicit question without '?' or question words."""
    _ensure_initialized()
    result = faq.search("tell me about the pay", language="en")
    assert result is not None
    assert result["category"] == "compensation"


def test_search_typos():
    """Misspelled question should still match via embeddings."""
    _ensure_initialized()
    result = faq.search("hwo much do they pay", language="en")
    assert result is not None
    assert result["category"] == "compensation"


def test_search_plain_statement_no_match():
    """Normal screening answer should not trigger a FAQ match."""
    _ensure_initialized()
    result = faq.search("Sí, tengo licencia", language="es")
    assert result is None
