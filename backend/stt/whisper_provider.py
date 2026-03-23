"""Whisper-based STT provider using ``faster-whisper`` (CTranslate2).

Performs multi-attempt transcription with optional VAD retries and
forced-language fallbacks to maximize accuracy for short bilingual audio.
"""

import asyncio
import io
import re
from dataclasses import dataclass

from faster_whisper import WhisperModel

from backend.config import (
    STT_AUTO_LANGUAGE_THRESHOLD,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_MODEL_SIZE,
    STT_VAD_MIN_SILENCE_MS,
)
from backend.logging_config import logger
from backend.stt.base import STTProvider

_LANG_MAP = {"es": "es", "en": "en"}
_SUPPORTED_LANGUAGES = tuple(_LANG_MAP.keys())
_NON_LATIN_SCRIPT_RE = re.compile(
    r"[\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u4E00-\u9FFF]"
)
_HALLUCINATION_RE = re.compile(
    r"amara\.org|"
    r"subt[ií]tulos\s+(por|de)\s+la\s+comunidad|"
    r"thank\s*you\s+for\s+watching|"
    r"thanks\s+for\s+watching|"
    r"please\s+subscribe|"
    r"suscr[ií]bete|"
    r"like\s+and\s+subscribe|"
    r"subtitles?\s+by|"
    r"sub(scribe|titled)\s+by|"
    r"translated\s+by|"
    r"transcri(bed|ption)\s+by|"
    r"www\.\w+\.\w{2,4}",
    re.IGNORECASE,
)
_TOKEN_CLEAN_RE = re.compile(r"[^a-zA-Z0-9'_-]+")
_SPANISH_HINTS = {
    "hola",
    "buenas",
    "gracias",
    "me",
    "llamo",
    "nombre",
    "si",
    "claro",
    "tengo",
    "vivo",
    "quiero",
    "puedo",
}
_ENGLISH_HINTS = {
    "hello",
    "hi",
    "thanks",
    "my",
    "name",
    "yes",
    "have",
    "live",
    "want",
    "can",
}


@dataclass
class _Attempt:
    """Result of a single transcription attempt with scoring metadata."""

    text: str
    detected_language: str
    language_probability: float | None
    avg_logprob: float | None
    avg_no_speech_prob: float | None
    score: float
    forced_language: str | None
    used_vad: bool
    duration: float | None


class WhisperSTTProvider(STTProvider):
    """STT provider using faster-whisper (CTranslate2 Whisper)."""

    def __init__(self) -> None:
        """Load the configured Whisper model into memory."""
        logger.info(
            "Loading faster-whisper model '{}' on {} ({})",
            STT_MODEL_SIZE,
            STT_DEVICE,
            STT_COMPUTE_TYPE,
        )
        self._model = WhisperModel(
            STT_MODEL_SIZE,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
        )
        logger.success("Whisper model loaded")

    async def transcribe(self, audio_bytes: bytes, language: str) -> tuple[str, str]:
        """Transcribe audio with auto-detection and forced-language fallbacks.

        Args:
            audio_bytes: Binary audio payload from the client.
            language: Requested language hint such as ``"es"`` or ``"en"``.

        Returns:
            Tuple of ``(best_transcript, detected_language)`` selected from the
            scored transcription attempts.
        """
        requested_language = _LANG_MAP.get(language, "es")
        attempts: list[_Attempt] = []

        auto_attempt = await self._transcribe_with_optional_vad_retry(
            audio_bytes,
            requested_language=requested_language,
            forced_language=None,
        )
        attempts.append(auto_attempt)

        if self._should_retry_with_forced_languages(auto_attempt):
            logger.info(
                "Auto STT detection looked unreliable "
                "(lang={} prob={} text_len={} score={:.2f}) - trying forced es/en",
                auto_attempt.detected_language,
                (
                    f"{auto_attempt.language_probability:.2f}"
                    if auto_attempt.language_probability is not None
                    else "n/a"
                ),
                len(auto_attempt.text),
                auto_attempt.score,
            )
            for candidate in self._candidate_languages(requested_language):
                attempts.append(
                    await self._transcribe_with_optional_vad_retry(
                        audio_bytes,
                        requested_language=requested_language,
                        forced_language=candidate,
                    )
                )

        best_attempt = max(attempts, key=lambda attempt: attempt.score)
        logger.debug(
            "Whisper selected transcript ({}, forced={}, prob={}, vad={}): '{}'",
            best_attempt.detected_language,
            best_attempt.forced_language or "auto",
            (
                f"{best_attempt.language_probability:.2f}"
                if best_attempt.language_probability is not None
                else "n/a"
            ),
            best_attempt.used_vad,
            best_attempt.text[:80],
        )
        return best_attempt.text, best_attempt.detected_language

    def supported_languages(self) -> list[str]:
        """Return the language codes supported by the fallback strategy.

        Returns:
            Supported language codes that can be auto-detected or forced.
        """
        return list(_SUPPORTED_LANGUAGES)

    def _candidate_languages(self, requested_language: str) -> list[str]:
        """Return supported languages ordered with the requested language first."""
        ordered = [requested_language]
        ordered.extend(
            language
            for language in _SUPPORTED_LANGUAGES
            if language != requested_language
        )
        return ordered

    async def _transcribe_with_optional_vad_retry(
        self,
        audio_bytes: bytes,
        *,
        requested_language: str,
        forced_language: str | None,
    ) -> _Attempt:
        """Run a transcription attempt and retry once without VAD if needed.

        Args:
            audio_bytes: Binary audio payload to transcribe.
            requested_language: Primary language hint from the caller.
            forced_language: Explicit language override for this attempt, or
                ``None`` to let Whisper auto-detect.

        Returns:
            The higher-scoring attempt between the VAD and non-VAD passes.
        """
        first_attempt = await self._transcribe_once(
            audio_bytes,
            requested_language=requested_language,
            forced_language=forced_language,
            use_vad=True,
        )
        if not self._should_retry_without_vad(first_attempt):
            return first_attempt

        logger.info(
            "Retrying STT without server-side VAD (forced_lang={}, detected_lang={}, text_len={})",
            forced_language or "auto",
            first_attempt.detected_language,
            len(first_attempt.text),
        )
        second_attempt = await self._transcribe_once(
            audio_bytes,
            requested_language=requested_language,
            forced_language=forced_language,
            use_vad=False,
        )
        return max([first_attempt, second_attempt], key=lambda attempt: attempt.score)

    async def _transcribe_once(
        self,
        audio_bytes: bytes,
        *,
        requested_language: str,
        forced_language: str | None,
        use_vad: bool,
    ) -> _Attempt:
        """Run one Whisper transcription request and score the result.

        Args:
            audio_bytes: Binary audio payload to transcribe.
            requested_language: Primary language hint from the caller.
            forced_language: Explicit language override for this attempt, or
                ``None`` for auto-detection.
            use_vad: Whether to enable server-side voice activity detection.

        Returns:
            Scored transcription attempt metadata used to select the best
            overall result.
        """
        audio_file = io.BytesIO(audio_bytes)
        kwargs = {
            "beam_size": 5,
            "condition_on_previous_text": False,
            "vad_filter": use_vad,
        }
        if use_vad:
            kwargs["vad_parameters"] = dict(
                min_silence_duration_ms=STT_VAD_MIN_SILENCE_MS
            )
        if forced_language:
            kwargs["language"] = forced_language

        segments, info = await asyncio.to_thread(
            self._model.transcribe,
            audio_file,
            **kwargs,
        )
        segment_list = list(segments)
        text = " ".join(
            seg.text.strip() for seg in segment_list if getattr(seg, "text", "").strip()
        ).strip()

        detected_language = (
            getattr(info, "language", None) or forced_language or requested_language
        )
        language_probability = getattr(info, "language_probability", None)
        avg_logprob = self._average_metric(segment_list, "avg_logprob")
        avg_no_speech_prob = self._average_metric(segment_list, "no_speech_prob")
        duration = getattr(info, "duration", None)
        if duration is None and segment_list:
            duration = max(
                (getattr(s, "end", 0) for s in segment_list), default=None
            )
        score = self._score_attempt(
            text=text,
            requested_language=requested_language,
            forced_language=forced_language,
            detected_language=detected_language,
            language_probability=language_probability,
            avg_logprob=avg_logprob,
            avg_no_speech_prob=avg_no_speech_prob,
            duration=duration,
        )

        logger.debug(
            "STT attempt forced={} vad={} detected={} prob={} score={:.2f} text='{}'",
            forced_language or "auto",
            use_vad,
            detected_language,
            (
                f"{language_probability:.2f}"
                if language_probability is not None
                else "n/a"
            ),
            score,
            text[:80],
        )
        return _Attempt(
            text=text,
            detected_language=detected_language,
            language_probability=language_probability,
            avg_logprob=avg_logprob,
            avg_no_speech_prob=avg_no_speech_prob,
            score=score,
            forced_language=forced_language,
            used_vad=use_vad,
            duration=duration,
        )

    def _should_retry_with_forced_languages(self, attempt: _Attempt) -> bool:
        """Decide whether the auto-detect result is unreliable enough to retry."""
        if not attempt.text:
            return False  # VAD found no speech — trust it, don't hallucinate
        if self._contains_non_latin_script(attempt.text):
            return True
        if attempt.detected_language not in _SUPPORTED_LANGUAGES:
            return True
        return (
            attempt.language_probability is not None
            and attempt.language_probability < STT_AUTO_LANGUAGE_THRESHOLD
        )

    def _should_retry_without_vad(self, attempt: _Attempt) -> bool:
        """Decide whether to retry without VAD filtering."""
        if not attempt.used_vad:
            return False
        # if not attempt.text:
        #     return True
        return (
            attempt.language_probability is not None
            and attempt.language_probability < 0.35
            and self._contains_non_latin_script(attempt.text)
        )

    def _score_attempt(
        self,
        *,
        text: str,
        requested_language: str,
        forced_language: str | None,
        detected_language: str,
        language_probability: float | None,
        avg_logprob: float | None,
        avg_no_speech_prob: float | None,
        duration: float | None,
    ) -> float:
        """Compute a heuristic quality score for a transcription attempt.

        Args:
            text: Transcript text produced by Whisper.
            requested_language: Primary language hint from the caller.
            forced_language: Explicit language override for this attempt, if
                any.
            detected_language: Language reported by Whisper.
            language_probability: Confidence score reported by Whisper.
            avg_logprob: Mean token log-probability across segments.
            avg_no_speech_prob: Mean no-speech probability across segments.
            duration: Audio duration in seconds.

        Returns:
            Composite score where higher values indicate a more trustworthy
            transcript.
        """
        if not text:
            return -100.0

        # P5: Sub-second audio cannot contain 10+ words — hallucination
        if duration is not None and duration < 1.0 and len(text.split()) > 10:
            return -100.0

        words = self._normalized_words(text)
        score = min(len(text), 80) * 0.04
        score += min(len(words), 12) * 0.35

        if self._contains_non_latin_script(text):
            score -= 12.0
        else:
            score += 2.0

        # P3: Reject when Whisper itself thinks the segment is not speech
        if avg_no_speech_prob is not None and avg_no_speech_prob > 0.7:
            return -100.0

        # P7: Combined low-confidence + probable non-speech → hallucination
        if (
            avg_logprob is not None
            and avg_logprob < -1.5
            and avg_no_speech_prob is not None
            and avg_no_speech_prob > 0.5
        ):
            return -100.0

        if avg_logprob is not None:
            score += max(min((avg_logprob + 2.5) * 2.0, 6.0), -6.0)
        if avg_no_speech_prob is not None:
            score += max(0.0, 1.0 - avg_no_speech_prob) * 1.5

        # P4: Reject known Whisper hallucination patterns
        if _HALLUCINATION_RE.search(text):
            return -100.0

        if forced_language == "es":
            score += self._keyword_bonus(words, _SPANISH_HINTS)
        elif forced_language == "en":
            score += self._keyword_bonus(words, _ENGLISH_HINTS)
        elif detected_language == "es":
            score += self._keyword_bonus(words, _SPANISH_HINTS) * 0.8
        elif detected_language == "en":
            score += self._keyword_bonus(words, _ENGLISH_HINTS) * 0.8

        if forced_language:
            if forced_language == requested_language:
                score += 0.4
        else:
            if (
                detected_language in _SUPPORTED_LANGUAGES
                and language_probability is not None
            ):
                score += language_probability * 4.0
            elif detected_language not in _SUPPORTED_LANGUAGES:
                score -= 3.0

        return score

    def _average_metric(self, segments: list, attr: str) -> float | None:
        """Compute the mean of a numeric segment attribute, or ``None`` if empty."""
        values = [
            value
            for value in (getattr(segment, attr, None) for segment in segments)
            if isinstance(value, (int, float))
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def _normalized_words(self, text: str) -> set[str]:
        """Lowercase and strip punctuation, returning a set of cleaned tokens."""
        words: set[str] = set()
        for raw_word in text.lower().split():
            word = _TOKEN_CLEAN_RE.sub("", raw_word)
            if word:
                words.add(word)
        return words

    def _keyword_bonus(self, words: set[str], hints: set[str]) -> float:
        """Score bonus for each recognized language-hint keyword found in the text."""
        return float(len(words & hints)) * 0.9

    def _contains_non_latin_script(self, text: str) -> bool:
        """Return True if the text contains non-Latin script (CJK, Cyrillic, etc.)."""
        return bool(_NON_LATIN_SCRIPT_RE.search(text))
