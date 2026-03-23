"""
Guardrails module — detects exit intent, offensive content, and confirmation/denial.

Architecture note:
    All detection functions use keyword/regex matching (zero-cost, deterministic).
    In a production environment with premium LLMs, these functions can be swapped
    to an LLM-as-Judge pattern (e.g., Claude Haiku / GPT-4o-mini classification)
    without changing the rest of the pipeline. The function signatures stay the same.
"""

import re
import unicodedata

from backend.logging_config import logger

# ---------------------------------------------------------------------------
# Exit-intent phrases (bilingual)
# ---------------------------------------------------------------------------

_EXIT_PHRASES_EN = {
    "i want to stop",
    "i don't want to continue",
    "i dont want to continue",
    "i do not want to continue",
    "i want to leave",
    "i'm done",
    "im done",
    "no thanks",
    "no thank you",
    "cancel",
    "quit",
    "stop",
    "bye",
    "goodbye",
    "good bye",
    "end",
    "leave",
    "exit",
    "i give up",
    "forget it",
    "never mind",
    "nevermind",
}

_EXIT_PHRASES_ES = {
    "quiero parar",
    "no quiero continuar",
    "no quiero seguir",
    "me quiero ir",
    "ya no quiero",
    "quiero salir",
    "no gracias",
    "parar",
    "salir",
    "terminar",
    "cancelar",
    "adiós",
    "adios",
    "chao",
    "chau",
    "me voy",
    "dejalo",
    "déjalo",
    "no me interesa",
    "ya no me interesa",
    "olvídalo",
    "olvidalo",
}

_ALL_EXIT_PHRASES = _EXIT_PHRASES_EN | _EXIT_PHRASES_ES

# Frontend explicit stop signal (sent by the Stop button)
FRONTEND_STOP_SIGNAL = "[STOP]"

# ---------------------------------------------------------------------------
# Confirmation / denial phrases
# ---------------------------------------------------------------------------

_CONFIRMATION_PHRASES = {
    # English
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "of course",
    "absolutely", "correct", "right", "affirmative", "definitely",
    # Spanish
    "si", "sí", "claro", "dale", "por supuesto", "correcto",
    "exacto", "vale", "de acuerdo", "seguro", "obvio",
}

_DENIAL_PHRASES = {
    # English
    "no", "nope", "nah", "not really", "no way", "never",
    "i changed my mind", "actually no",
    # Spanish
    "no", "nop", "nel", "para nada", "no quiero",
    "cambié de opinión", "cambie de opinion", "la verdad no",
}

# ---------------------------------------------------------------------------
# Offensive content — curated word-boundary patterns (EN + ES)
# ---------------------------------------------------------------------------
# Conservative list focused on unambiguous slurs/profanity.
# Each entry is matched with \b word boundaries to avoid false positives.

_OFFENSIVE_WORDS_EN = [
    "fuck", "fucking", "fucker", "fucked", "fck",
    "shit", "shitty", "bullshit",
    "asshole", "arsehole",
    "bitch", "bitches",
    "damn", "damned",
    "bastard", "bastards",
    "dick", "dickhead",
    "cunt",
    "motherfucker", "mf",
    "stfu", "gtfo",
    "idiot", "moron", "retard", "retarded",
    "stupid bot", "dumb bot", "useless bot",
    "piece of shit", "pos",
    "go to hell", "screw you", "f you", "f off",
    "kys",
]

_OFFENSIVE_WORDS_ES = [
    "mierda", "mierdas",
    "puta", "putas", "puto", "putos",
    "hijo de puta", "hijueputa", "hdp",
    "pendejo", "pendeja", "pendejos",
    "idiota", "idiotas",
    "estúpido", "estupido", "estúpida", "estupida",
    "imbécil", "imbecil",
    "cabrón", "cabron", "cabrona",
    "chinga", "chingada", "chingar", "chinga tu madre",
    "verga", "a la verga",
    "culero", "culera",
    "pinche",
    "mamón", "mamon",
    "joder", "jódete", "jodete",
    "gilipollas",
    "coño",
    "vete a la mierda",
    "vete al diablo",
    "carajo",
]

# Pre-compile a single regex for performance.
# Sort by length descending so multi-word phrases match before their parts.
_all_offensive = sorted(
    _OFFENSIVE_WORDS_EN + _OFFENSIVE_WORDS_ES,
    key=len,
    reverse=True,
)
_OFFENSIVE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _all_offensive) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Bilingual response messages
# ---------------------------------------------------------------------------

_EXIT_CONFIRMATION = {
    "es": (
        "Parece que quieres terminar la conversación. "
        "¿Estás seguro/a de que deseas salir del proceso de selección?"
    ),
    "en": (
        "It looks like you'd like to end the conversation. "
        "Are you sure you want to stop the screening process?"
    ),
}

_WITHDRAWAL_MESSAGE = {
    "es": (
        "Entendido{name_part}. Hemos terminado la conversación. "
        "Si cambias de opinión en el futuro, no dudes en volver a contactarnos. "
        "¡Te deseamos lo mejor!"
    ),
    "en": (
        "Understood{name_part}. We've ended the conversation. "
        "If you change your mind in the future, don't hesitate to reach out again. "
        "We wish you all the best!"
    ),
}

_OFFENSIVE_WARNING = {
    "es": (
        "Entiendo que puedas estar frustrado/a, pero te pido que mantengamos "
        "un tono respetuoso para poder continuar con el proceso. "
        "¿Continuamos con la pregunta anterior?"
    ),
    "en": (
        "I understand you might be frustrated, but I'd appreciate it if we could "
        "keep things respectful so we can continue the process. "
        "Shall we continue with the previous question?"
    ),
}

_OFFENSIVE_TERMINATION = {
    "es": (
        "Lamentablemente, no podemos continuar con el proceso de selección "
        "debido al tono de la conversación. Si deseas volver a aplicar "
        "en el futuro, eres bienvenido/a. ¡Te deseamos lo mejor!"
    ),
    "en": (
        "Unfortunately, we're unable to continue the screening process "
        "due to the tone of the conversation. If you'd like to reapply "
        "in the future, you're welcome to do so. We wish you all the best!"
    ),
}

# ---------------------------------------------------------------------------
# Public detection functions
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation (except apostrophes), and collapse whitespace.

    Args:
        text: Raw user input.

    Returns:
        Cleaned text suitable for phrase-set lookups.
    """
    text = text.strip().lower()
    # Remove punctuation except apostrophes (for "don't", "i'm", etc.)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_exit_intent(text: str) -> bool:
    """Detect whether the message expresses intent to stop the conversation.

    Only triggers on short, unambiguous messages to avoid false positives
    on longer answers that happen to contain exit-like words (e.g.
    "I quit my last job and went to Deliveroo").

    Args:
        text: Raw user message.

    Returns:
        ``True`` if an exit intent is detected.
    """
    if text.strip() == FRONTEND_STOP_SIGNAL:
        return True

    normalized = _normalize(text)

    # Only match short messages (≤ 6 words) to avoid false positives on
    # longer conversational answers that happen to contain exit keywords.
    if len(normalized.split()) > 6:
        return False

    if normalized in _ALL_EXIT_PHRASES:
        return True

    # Check if the message starts with a known phrase followed only by
    # filler words (e.g. "stop please", "bye bye", "quit now").
    _TRAILING_FILLER = {"please", "now", "bye", "thanks", "por favor", "ya", "gracias"}
    for phrase in _ALL_EXIT_PHRASES:
        if normalized.startswith(phrase):
            remainder = normalized[len(phrase):].strip()
            # Only match if nothing follows, or only filler words follow
            if not remainder or all(w in _TRAILING_FILLER for w in remainder.split()):
                return True

    return False


def detect_offensive_content(text: str) -> bool:
    """Check whether the message contains offensive language.

    Uses a pre-compiled regex of bilingual (EN/ES) profanity patterns
    with word-boundary matching to minimise false positives.

    Args:
        text: Raw user message.

    Returns:
        ``True`` if offensive content is detected.
    """
    return bool(_OFFENSIVE_PATTERN.search(text))


def detect_confirmation(text: str) -> bool:
    """Check whether the message is an affirmative confirmation (e.g. "yes", "sí").

    Args:
        text: Raw user message.

    Returns:
        ``True`` if the normalised text matches a known confirmation phrase.
    """
    normalized = _normalize(text)
    return normalized in _CONFIRMATION_PHRASES


def detect_denial(text: str) -> bool:
    """Check whether the message is a negative denial (e.g. "no", "nah").

    Args:
        text: Raw user message.

    Returns:
        ``True`` if the normalised text matches a known denial phrase.
    """
    normalized = _normalize(text)
    return normalized in _DENIAL_PHRASES


# ---------------------------------------------------------------------------
# Public message getters
# ---------------------------------------------------------------------------


def get_exit_confirmation_message(language: str) -> str:
    """Return the "are you sure you want to stop?" prompt.

    Args:
        language: ``"es"`` or ``"en"``.

    Returns:
        Localised confirmation prompt string.
    """
    return _EXIT_CONFIRMATION.get(language, _EXIT_CONFIRMATION["es"])


def get_withdrawal_message(language: str, name: str | None = None) -> str:
    """Return the farewell message for a voluntary withdrawal.

    Args:
        language: ``"es"`` or ``"en"``.
        name: Candidate's name to personalise the message, or ``None``.

    Returns:
        Localised farewell string.
    """
    name_part = f", {name}" if name else ""
    template = _WITHDRAWAL_MESSAGE.get(language, _WITHDRAWAL_MESSAGE["es"])
    return template.format(name_part=name_part)


def get_offensive_warning_message(language: str) -> str:
    """Return the first-strike warning for offensive language.

    Args:
        language: ``"es"`` or ``"en"``.

    Returns:
        Localised warning string.
    """
    return _OFFENSIVE_WARNING.get(language, _OFFENSIVE_WARNING["es"])


def get_offensive_termination_message(language: str) -> str:
    """Return the termination message after repeated offenses.

    Args:
        language: ``"es"`` or ``"en"``.

    Returns:
        Localised termination string.
    """
    return _OFFENSIVE_TERMINATION.get(language, _OFFENSIVE_TERMINATION["es"])
