"""Edge TTS provider using Microsoft's free online neural TTS service."""

import io

import edge_tts

from backend.tts.base import TTSProvider
from backend.config import TTS_EDGE_VOICE_ES, TTS_EDGE_VOICE_EN


# Default voice mapping per language
_VOICE_MAP = {
    "es": TTS_EDGE_VOICE_ES,
    "en": TTS_EDGE_VOICE_EN,
}


class EdgeTTSProvider(TTSProvider):
    """TTS provider using Microsoft Edge's free online TTS service via edge-tts."""

    async def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        """Synthesize MP3 audio using an Edge neural voice.

        Args:
            text: Text to convert into speech.
            language: Requested language code used to choose the default voice.
            voice: Optional explicit Edge voice identifier.

        Returns:
            MP3 audio bytes streamed back by ``edge-tts``.

        Raises:
            RuntimeError: If the provider returns no audio data.
        """
        selected_voice = voice or _VOICE_MAP.get(language, _VOICE_MAP["es"])
        communicate = edge_tts.Communicate(text, selected_voice)

        buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])

        audio_bytes = buffer.getvalue()
        if not audio_bytes:
            raise RuntimeError("edge-tts returned empty audio")
        return audio_bytes

    def audio_content_type(self) -> str:
        """Return the MIME type produced by Edge TTS.

        Returns:
            ``"audio/mpeg"`` for the MP3 stream returned by ``edge-tts``.
        """
        return "audio/mpeg"

    async def list_voices(self, language: str) -> list[dict]:
        """List Edge voices for the requested language family.

        Args:
            language: Requested language code such as ``"es"`` or ``"en"``.

        Returns:
            Voice metadata records containing ID, display name, and gender.
        """
        all_voices = await edge_tts.list_voices()
        prefix = "es" if language == "es" else "en"
        return [
            {"id": v["ShortName"], "name": v["FriendlyName"], "gender": v["Gender"]}
            for v in all_voices
            if v["Locale"].startswith(prefix)
        ]
