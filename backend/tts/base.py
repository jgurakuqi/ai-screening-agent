"""Interfaces shared by all text-to-speech providers."""

from abc import ABC, abstractmethod


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers."""

    @abstractmethod
    async def synthesize(self, text: str, language: str, voice: str | None = None) -> bytes:
        """Synthesize speech from text.

        Args:
            text: Text to convert into audio.
            language: Requested language code such as ``"es"`` or ``"en"``.
            voice: Optional provider-specific voice identifier.

        Returns:
            Encoded audio bytes produced by the provider.
        """

    @abstractmethod
    def audio_content_type(self) -> str:
        """Return the MIME type used for the provider's audio output.

        Returns:
            MIME type string such as ``"audio/mpeg"``.
        """

    @abstractmethod
    async def list_voices(self, language: str) -> list[dict]:
        """List available voices for a given language.

        Args:
            language: Requested language code such as ``"es"`` or ``"en"``.

        Returns:
            Voice metadata records understood by the caller.
        """
