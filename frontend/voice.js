/**
 * @file Voice Mode Module — cross-browser STT and TTS for the screening chatbot.
 *
 * Provides:
 * - {@link SpeechRecognizer}: MediaRecorder + server-side faster-whisper STT
 *   with Web Audio API silence detection for automatic stop.
 * - {@link SpeechSynthesizer}: Server-side edge-tts with browser SpeechSynthesis fallback.
 * - {@link VoiceMode}: Orchestrator that chains listen → transcribe → send → speak → listen.
 */

const API_BASE = window.location.origin;

/**
 * Speech-to-text via MediaRecorder + server-side Whisper transcription.
 *
 * Uses a Web Audio API AnalyserNode for volume-based silence detection so
 * recording auto-stops when the user finishes speaking.
 *
 * The mic stream is acquired fresh on each {@link start} and fully released
 * (all tracks stopped) on each {@link stop}. This avoids Windows
 * "Communications Activity" audio ducking during TTS / idle.
 */
class SpeechRecognizer {
  // --- Silence-detection tuning ---
  static VAD_INTERVAL_MS = 100;   // how often we sample volume
  static SILENCE_THRESHOLD = 12;    // RMS below this = "silence" (0-128 scale)
  static SILENCE_DURATION_MS = 1500;  // silence this long after speech → auto-stop
  static MIN_SPEECH_DURATION_MS = 600;  // must detect speech for this long before silence-stop arms
  static MAX_RECORD_MS = 30000; // hard cap — stop after 30 s no matter what

  /**
   * @param {object} callbacks
   * @param {(text: string, detectedLang: string) => void} callbacks.onResult - Final transcript.
   * @param {(text: string) => void} callbacks.onInterim - Interim status text.
   * @param {(error: string) => void} callbacks.onError - Error code (e.g. "not-allowed").
   * @param {() => void} callbacks.onEnd - Fired when recognition cycle finishes.
   */
  constructor({ onResult, onInterim, onError, onEnd }) {
    this.supported = !!(
      navigator.mediaDevices?.getUserMedia &&
      typeof MediaRecorder !== "undefined"
    );

    this._listening = false;
    this._onResult = onResult;
    this._onInterim = onInterim;
    this._onError = onError;
    this._onEnd = onEnd;

    // All per-cycle — created in start(), torn down in _releaseStream()
    this._stream = null;
    this._recorder = null;
    this._chunks = [];
    this._language = "es";
    this._audioCtx = null;
    this._analyser = null;
    this._sourceNode = null;

    // Silence-detection state
    this._vadTimer = null;
    this._maxTimer = null;
    this._speechStart = 0;
    this._silenceStart = 0;
    this._hasSpeech = false;
  }

  // ---- Public API ----

  /**
   * Acquire the microphone, start recording, and begin silence detection.
   * @param {string} [language="es"] - Language hint for server-side Whisper.
   */
  async start(language) {
    if (!this.supported || this._listening) {
      console.log("[STT] start() skipped — supported:", this.supported, "listening:", this._listening);
      return;
    }

    this._language = language || "es";
    console.log("[STT] start() called, lang:", this._language);

    // ---- Acquire a fresh mic stream every cycle ----
    try {
      console.log("[STT] requesting microphone access...");
      this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log("[STT] microphone access granted, tracks:", this._stream.getAudioTracks().length);
    } catch (e) {
      console.warn("[STT] getUserMedia failed:", e.name, e.message);
      if (this._onError) this._onError("not-allowed");
      return;
    }

    // ---- Set up Web Audio analyser for silence detection ----
    const ACtor = window.AudioContext || window.webkitAudioContext;
    this._audioCtx = new ACtor();
    this._sourceNode = this._audioCtx.createMediaStreamSource(this._stream);
    this._analyser = this._audioCtx.createAnalyser();
    this._analyser.fftSize = 512;
    this._sourceNode.connect(this._analyser);
    // Do NOT connect analyser to destination — we don't want to play mic audio back
    console.log("[STT] AudioContext created, sampleRate:", this._audioCtx.sampleRate);

    // ---- Set up MediaRecorder ----
    let mimeType = "";
    if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
      mimeType = "audio/webm;codecs=opus";
    }
    this._chunks = [];
    const options = mimeType ? { mimeType } : {};
    this._recorder = new MediaRecorder(this._stream, options);
    console.log("[STT] MediaRecorder created, mimeType:", this._recorder.mimeType);

    this._recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this._chunks.push(e.data);
    };

    this._recorder.onstop = () => {
      console.log("[STT] MediaRecorder onstop fired, chunks:", this._chunks.length);
      this._handleRecordingStop();
    };

    this._recorder.onerror = (e) => {
      console.error("[STT] MediaRecorder error:", e.error?.name, e.error?.message);
    };

    // ---- Start recording + silence detection ----
    this._listening = true;
    this._hasSpeech = false;
    this._speechStart = 0;
    this._silenceStart = 0;

    if (this._onInterim) this._onInterim("\u{1F3A4} Recording...");
    this._recorder.start();
    this._startVAD();

    // Hard safety cap
    this._maxTimer = setTimeout(() => {
      console.log("[STT] max recording duration reached, auto-stopping");
      this.stop();
    }, SpeechRecognizer.MAX_RECORD_MS);

    console.log("[STT] \u25B6 recording started, lang:", this._language, "mime:", this._recorder.mimeType);
  }

  /** Stop recording, release the mic, and trigger server-side transcription. */
  stop() {
    console.log("[STT] stop() called — listening:", this._listening, "recorder state:", this._recorder?.state);
    if (!this._listening || !this._recorder) {
      console.log("[STT] stop() skipped — nothing to stop");
      return;
    }
    this._listening = false;
    this._stopVAD();

    if (this._recorder.state === "recording") {
      console.log("[STT] \u23F9 stopping MediaRecorder...");
      this._recorder.stop(); // triggers onstop → _handleRecordingStop
    } else {
      console.warn("[STT] recorder not in 'recording' state:", this._recorder.state);
      // Still release resources even if recorder wasn't recording
      this._releaseStream();
    }
  }

  /** @returns {boolean} True if currently recording. */
  isListening() {
    return this._listening;
  }

  /** Release everything. Call from VoiceMode.disable(). */
  destroy() {
    this.stop();
    this._releaseStream();
  }

  // ---- Resource management ----

  /** Release mic stream, AudioContext, and all audio nodes.
   *  This ends Windows "Communications Activity" ducking immediately. */
  _releaseStream() {
    if (this._sourceNode) {
      try { this._sourceNode.disconnect(); } catch (_) { /* ignore */ }
      this._sourceNode = null;
    }
    this._analyser = null;
    if (this._audioCtx && this._audioCtx.state !== "closed") {
      this._audioCtx.close().catch(() => { });
      this._audioCtx = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
      console.log("[STT] mic stream released (all tracks stopped)");
    }
  }

  // ---- Silence detection via Web Audio API ----

  /** Start polling the analyser node for volume-based silence detection. */
  _startVAD() {
    const bufferLen = this._analyser.fftSize;
    const dataArray = new Uint8Array(bufferLen);

    this._vadTimer = setInterval(() => {
      if (!this._analyser) return; // guard against late ticks after teardown
      this._analyser.getByteTimeDomainData(dataArray);

      // Compute RMS volume (0-128 scale, where 0 = silence)
      let sumSq = 0;
      for (let i = 0; i < bufferLen; i++) {
        const v = dataArray[i] - 128; // center around 0
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / bufferLen);

      const now = Date.now();
      const isSilent = rms < SpeechRecognizer.SILENCE_THRESHOLD;

      if (!isSilent) {
        // Sound detected
        if (!this._hasSpeech) {
          this._hasSpeech = true;
          this._speechStart = now;
          console.log("[VAD] speech detected (RMS:", rms.toFixed(1), ")");
          if (this._onInterim) this._onInterim("\u{1F3A4} Listening...");
        }
        this._silenceStart = 0; // reset silence timer
      } else if (this._hasSpeech) {
        // Silence after speech
        if (this._silenceStart === 0) {
          this._silenceStart = now;
        }
        const speechDuration = now - this._speechStart;
        const silenceDuration = now - this._silenceStart;

        if (
          speechDuration >= SpeechRecognizer.MIN_SPEECH_DURATION_MS &&
          silenceDuration >= SpeechRecognizer.SILENCE_DURATION_MS
        ) {
          console.log("[VAD] silence detected after", speechDuration, "ms of speech — auto-stopping");
          this.stop();
        }
      }
    }, SpeechRecognizer.VAD_INTERVAL_MS);
  }

  /** Stop the silence detection interval and the max-duration safety timer. */
  _stopVAD() {
    if (this._vadTimer) {
      clearInterval(this._vadTimer);
      this._vadTimer = null;
    }
    if (this._maxTimer) {
      clearTimeout(this._maxTimer);
      this._maxTimer = null;
    }
  }

  // ---- Transcription ----

  /** Send the recorded audio blob to the server for Whisper transcription. */
  async _handleRecordingStop() {
    const mimeType = this._recorder?.mimeType || "audio/webm";
    const blob = new Blob(this._chunks, { type: mimeType });
    const totalChunks = this._chunks.length;
    this._chunks = [];
    this._recorder = null;

    // Release mic + AudioContext immediately so OS audio ducking ends NOW,
    // before we even send the network request for transcription.
    this._releaseStream();

    console.log("[STT] recording stopped — blob size:", blob.size, "bytes, chunks:", totalChunks, "mimeType:", mimeType);

    // Very small blobs are likely silence — skip transcription
    if (blob.size < 1000) {
      console.log("[STT] recording too small (< 1000 bytes), skipping transcription");
      if (this._onEnd) this._onEnd();
      return;
    }

    // No speech detected by client-side VAD — skip transcription to prevent
    // Whisper from hallucinating text from silence/noise.
    if (!this._hasSpeech) {
      console.log("[STT] no speech detected during recording, skipping transcription");
      if (this._onEnd) this._onEnd();
      return;
    }

    // Show transcribing feedback
    if (this._onInterim) this._onInterim("Transcribing...");
    console.log("[STT] sending audio to server for transcription...");

    const ext = blob.type.includes("mp4") ? "recording.mp4" : "recording.webm";
    const form = new FormData();
    form.append("audio", blob, ext);
    form.append("language", this._language);

    try {
      const t0 = performance.now();
      const res = await fetch(`${API_BASE}/stt/transcribe`, {
        method: "POST",
        body: form,
      });
      const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
      console.log("[STT] server responded in", elapsed, "s — status:", res.status);

      if (!res.ok) {
        const errText = await res.text();
        console.error("[STT] server error response:", errText);
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      console.log("[STT] server response:", JSON.stringify(data));

      if (data.empty || !data.transcript?.trim()) {
        console.log("[STT] empty transcript — user said nothing");
        if (this._onEnd) this._onEnd();
        return;
      }

      console.log("[STT] \u2705 transcript:", data.transcript, "detected_language:", data.detected_language);
      if (this._onResult) this._onResult(data.transcript, data.detected_language);
      if (this._onEnd) this._onEnd();
    } catch (e) {
      console.error("[STT] \u274C transcription failed:", e.message, e);
      if (this._onError) this._onError("server-error");
      if (this._onEnd) this._onEnd();
    }
  }
}

/**
 * Text-to-speech synthesizer with server-side edge-tts and browser fallback.
 *
 * Uses a generation counter to safely cancel stale playback when a new
 * {@link speak} call arrives before the previous one finishes.
 */
class SpeechSynthesizer {
  constructor() {
    this._speaking = false;
    this._onEnd = null;
    this._generation = 0;
  }

  /**
   * Synthesize and play speech for the given text.
   * @param {string} text - Text to speak.
   * @param {string} language - Language code ("es" or "en").
   * @param {(() => void)|null} [onEnd] - Callback when playback finishes.
   */
  async speak(text, language, onEnd) {
    this.stop();
    this._generation++;
    const gen = this._generation;
    this._onEnd = onEnd || null;

    try {
      const res = await fetch(`${API_BASE}/tts/synthesize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language }),
      });

      if (gen !== this._generation) return;

      const contentType = res.headers.get("content-type") || "";

      if (contentType.includes("application/json")) {
        const data = await res.json();
        if (data.use_browser_tts) {
          this._browserSpeak(text, language, gen);
          return;
        }
      }

      if (!res.ok) {
        throw new Error(`TTS server error: ${res.status}`);
      }

      const blob = await res.blob();
      if (gen !== this._generation) return;

      // Play via blob URL. The key: do NOT await play().
      // Chrome can reject the play() promise even when audio plays fine
      // (e.g., "media removed from document"). Instead we rely on onended.
      const url = URL.createObjectURL(blob);
      const audio = new Audio();
      this._audio = audio;
      this._speaking = true;

      audio.onended = () => {
        URL.revokeObjectURL(url);
        if (gen !== this._generation) return;
        console.log("[TTS] edge-tts audio finished");
        this._speaking = false;
        if (this._onEnd) this._onEnd();
      };

      audio.src = url;
      console.log("[TTS] playing edge-tts audio,", blob.size, "bytes");

      // Fire-and-forget play — don't let a rejected promise trigger fallback
      audio.play().catch((e) => {
        // Only fall back if audio truly didn't start
        // Check if audio has progressed at all
        if (audio.currentTime === 0 && audio.paused) {
          console.warn("[TTS] play() failed, audio did not start:", e.message);
          URL.revokeObjectURL(url);
          if (gen !== this._generation) return;
          this._browserSpeak(text, language, gen);
        } else {
          // Audio is actually playing despite the rejected promise — ignore
          console.log("[TTS] play() promise rejected but audio is playing, ignoring");
        }
      });
    } catch (e) {
      if (gen !== this._generation) return;
      console.warn("[TTS] fetch failed, falling back to browser:", e.message);
      this._browserSpeak(text, language, gen);
    }
  }

  /**
   * Fallback: use the browser's native SpeechSynthesis API.
   * @param {string} text
   * @param {string} language
   * @param {number} gen - Generation counter for cancellation safety.
   */
  _browserSpeak(text, language, gen) {
    if (!window.speechSynthesis) {
      this._speaking = false;
      if (gen === this._generation && this._onEnd) this._onEnd();
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language === "en" ? "en-US" : "es-ES";
    utterance.rate = 1.0;

    this._speaking = true;
    console.log("[TTS] using browser SpeechSynthesis fallback");

    utterance.onend = () => {
      if (gen !== this._generation) return;
      this._speaking = false;
      if (this._onEnd) this._onEnd();
    };

    utterance.onerror = () => {
      if (gen !== this._generation) return;
      this._speaking = false;
      if (this._onEnd) this._onEnd();
    };

    window.speechSynthesis.speak(utterance);
  }

  /** Stop any current playback and cancel pending speech. */
  stop() {
    this._generation++;
    this._speaking = false;
    this._onEnd = null;
    if (this._audio) {
      this._audio.pause();
      this._audio.onended = null;
      this._audio = null;
    }
    if (window.speechSynthesis) window.speechSynthesis.cancel();
  }

  /** @returns {boolean} True if audio is currently playing. */
  isSpeaking() {
    return this._speaking;
  }
}

/**
 * Voice Mode orchestrator — chains STT → send message → TTS → auto-listen.
 *
 * Exposed globally as `window.voiceMode` and driven by the voice toggle button.
 * The three-state toggle cycles: OFF → ON (listening) → stop recording → OFF.
 */
class VoiceMode {
  constructor() {
    this.enabled = false;
    this._language = "es";

    this.synthesizer = new SpeechSynthesizer();
    this.recognizer = new SpeechRecognizer({
      onResult: (text, detectedLang) => this._handleSpeechResult(text, detectedLang),
      onInterim: (text) => this._handleInterim(text),
      onError: (error) => this._handleSpeechError(error),
      onEnd: () => this._handleRecognitionEnd(),
    });

    this.supported = this.recognizer.supported;
    this._autoListenAfterTTS = true;
    this._waitingForTTS = false;
    this._pendingAgentReply = false; // true after we send a message, cleared by onAssistantMessage
  }

  /**
   * Set the language for STT recognition.
   * @param {string} lang - "es" or "en".
   */
  setLanguage(lang) {
    this._language = lang === "en" ? "en" : "es";
  }

  /**
   * Two-state toggle: OFF → ON (start listening) | ON → OFF (disable).
   * Recording stops automatically via silence detection (VAD).
   * @returns {boolean} Whether voice mode is enabled after the toggle.
   */
  toggle() {
    if (!this.enabled) {
      console.log("[Voice] toggle: OFF → ON (enable + start listening)");
      this.enable();
      this.startListening();
    } else {
      console.log("[Voice] toggle: ON → OFF (disable)");
      this.disable();
    }
    return this.enabled;
  }

  /** Activate voice mode (does not start listening — call {@link startListening}). */
  enable() {
    this.enabled = true;
    console.log("[Voice] voice mode enabled");
    const $input = document.getElementById("msg-input");
    if ($input) {
      $input.disabled = true;
      $input.placeholder = "Voice mode active — speak to respond";
    }
    this._updateUI();
  }

  /** Deactivate voice mode, stop recording, and cancel any TTS playback. */
  disable() {
    console.log("[Voice] voice mode disabled");
    this.enabled = false;
    this.recognizer.destroy();
    this.synthesizer.stop();
    const $input = document.getElementById("msg-input");
    if ($input) {
      $input.disabled = false;
      $input.value = "";
      $input.style.fontStyle = "normal";
      $input.placeholder = "Type your message...";
    }
    this._updateUI();
  }

  /** Begin STT recording if voice mode is active and supported. */
  startListening() {
    if (!this.enabled || !this.supported) {
      console.log("[Voice] startListening skipped — enabled:", this.enabled, "supported:", this.supported);
      return;
    }
    console.log("[Voice] startListening, lang:", this._language);
    this.recognizer.start(this._language);
    this._updateUI();
  }

  /** Stop STT recording (triggers transcription). */
  stopListening() {
    console.log("[Voice] stopListening called");
    this.recognizer.stop();
    this._updateUI();
  }

  /**
   * Called by app.js when the agent responds — speaks the reply and auto-listens after.
   * @param {string} text - Agent response text.
   * @param {string} language - User's language (for next STT cycle).
   * @param {string} [ttsLanguage] - Language for TTS voice (may differ from user language).
   */
  onAssistantMessage(text, language, ttsLanguage) {
    if (!this.enabled) return;
    this._pendingAgentReply = false;
    if (language) this._language = language === "en" ? "en" : "es";
    // TTS voice should match the response language, not the user's language
    const voiceLang = ttsLanguage || this._language;
    console.log("[Voice] onAssistantMessage — sttLang:", this._language, "ttsLang:", voiceLang, "text:", text.substring(0, 60) + "...");

    this.recognizer.stop();
    this._waitingForTTS = true;
    this._updateUI();

    // Safety: if TTS callback never fires (browser quirk), auto-listen after a timeout.
    // Estimate ~80ms per character as a rough TTS duration upper bound.
    const safetyMs = Math.max(10000, text.length * 80 + 3000);
    let ttsCallbackFired = false;

    const ttsSafetyTimer = setTimeout(() => {
      if (!ttsCallbackFired && this.enabled) {
        console.warn("[Voice] TTS safety timeout — callback never fired, forcing auto-listen");
        this._waitingForTTS = false;
        if (this._autoListenAfterTTS) {
          this.startListening();
        } else {
          this._updateUI();
        }
      }
    }, safetyMs);

    this.synthesizer.speak(text, voiceLang, () => {
      ttsCallbackFired = true;
      clearTimeout(ttsSafetyTimer);
      console.log("[Voice] TTS finished — autoListen:", this._autoListenAfterTTS, "enabled:", this.enabled);
      this._waitingForTTS = false;
      if (this.enabled && this._autoListenAfterTTS) {
        setTimeout(() => {
          if (this.enabled) {
            console.log("[Voice] auto-listen: starting listening after TTS");
            this.startListening();
          }
        }, 300);
      } else {
        this._updateUI();
      }
    });
  }

  /** Stop listening and disable auto-listen (conversation has ended). */
  stopForTerminal() {
    this._autoListenAfterTTS = false;
    this.recognizer.stop();
  }

  /** Re-enable automatic listening after TTS (e.g. for a new conversation). */
  resumeAutoListen() {
    this._autoListenAfterTTS = true;
  }

  /** Handle a final transcript from the recognizer — inject into input and send. */
  _handleSpeechResult(text, detectedLang) {
    if (!this.enabled || !text) {
      console.log("[Voice] _handleSpeechResult ignored — enabled:", this.enabled, "text:", text);
      return;
    }
    console.log("[Voice] \u2705 speech result:", text, "whisper_language:", detectedLang);

    // Store Whisper's detected language so sendMessage() can forward it to backend
    this.lastWhisperLanguage = detectedLang || null;

    // Mark that we're waiting for the agent to reply — prevents re-arm from
    // restarting the listener before onAssistantMessage fires.
    this._pendingAgentReply = true;

    // Temporarily enable and populate the input so sendMessage() can read it
    const $input = document.getElementById("msg-input");
    if ($input) {
      $input.disabled = false;
      $input.value = text;
      $input.style.fontStyle = "normal";
    }

    if (typeof window.sendMessage === "function") {
      window.sendMessage();
    }

    // Re-disable the input after sending (voice mode keeps it disabled)
    if ($input && this.enabled) {
      $input.disabled = true;
    }

    this._updateUI();
  }

  /** Show interim status text (e.g. "Recording…", "Listening…") in the voice indicator. */
  _handleInterim(text) {
    const $indicator = document.getElementById("voice-indicator");
    if ($indicator) {
      $indicator.textContent = text;
      $indicator.hidden = false;
      $indicator.className = "voice-indicator listening";
    }
  }

  /** Handle recognizer errors (e.g. microphone permission denied). */
  _handleSpeechError(error) {
    console.warn("[Voice] speech error:", error);

    if (error === "not-allowed") {
      this.disable();
      const $input = document.getElementById("msg-input");
      if ($input) {
        $input.placeholder = "Microphone access denied. Please allow microphone and try again.";
      }
    }
  }

  /** Re-arm the listener after an empty/silent recognition if still in voice mode. */
  _handleRecognitionEnd() {
    console.log("[Voice] recognition end — enabled:", this.enabled,
      "autoListen:", this._autoListenAfterTTS,
      "waitingForTTS:", this._waitingForTTS);
    const $input = document.getElementById("msg-input");
    if ($input) $input.style.fontStyle = "normal";
    this._updateUI();

    // If voice mode is still active and we're NOT waiting for the agent to
    // respond (i.e. the transcript was empty / silence), re-arm listening
    // so the user doesn't get stuck in a dead state.
    // Skip re-arm if we just sent a message (_pendingAgentReply) — the
    // onAssistantMessage → TTS → auto-listen chain handles that case.
    if (this.enabled && !this._waitingForTTS && !this._pendingAgentReply && !this.recognizer.isListening()) {
      console.log("[Voice] empty/silent result — re-arming listener in 500 ms");
      setTimeout(() => {
        if (this.enabled && !this._waitingForTTS && !this._pendingAgentReply && !this.recognizer.isListening()) {
          this.startListening();
        }
      }, 300);
    }
  }

  /** Sync the voice button and status indicator with the current voice state. */
  _updateUI() {
    const $btn = document.getElementById("btn-voice");
    if (!$btn) return;

    $btn.classList.toggle("active", this.enabled);
    $btn.classList.toggle("listening", this.recognizer.isListening());

    const $indicator = document.getElementById("voice-indicator");
    if ($indicator) {
      $indicator.hidden = !this.enabled;
      if (this.recognizer.isListening()) {
        $indicator.textContent = "Listening...";
        $indicator.className = "voice-indicator listening";
      } else if (this._waitingForTTS) {
        $indicator.textContent = "Speaking...";
        $indicator.className = "voice-indicator speaking";
      } else if (this.enabled) {
        $indicator.textContent = "Voice mode ON";
        $indicator.className = "voice-indicator";
      }
    }
  }
}

// ---- Static check & global export ----
(function () {
  const hasSTT = !!(
    navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined"
  );

  window.voiceMode = new VoiceMode();

  document.addEventListener("DOMContentLoaded", () => {
    const $btn = document.getElementById("btn-voice");
    if ($btn && hasSTT) {
      $btn.hidden = false;
      $btn.addEventListener("click", () => {
        window.voiceMode.toggle();
      });
    }
  });
})();
