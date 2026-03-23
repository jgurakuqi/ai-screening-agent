/**
 * @file Main frontend application for the Grupo Sazón screening chatbot.
 *
 * Manages the chat UI, sidebar (conversation list, analytics, resizing),
 * theme toggle, and communication with the FastAPI backend via REST calls.
 */

// ---- Configuration ----
const API = window.location.origin;

// ---- State ----

/** @type {{ conversationId: string|null, status: string|null, stage: string|null, messages: Array<{role: string, content: string}> }} */
const state = {
  conversationId: null,
  status: null,
  stage: null,
  messages: [],
};

/** AbortController for the current selectConversation fetch — cancelled on re-entry. */
let _selectConvController = null;

/** Interval ID for polling new messages (e.g. re-engagement nudges). */
let _pollIntervalId = null;
const POLL_INTERVAL_MS = 15_000; // check every 15 seconds

// ---- DOM refs ----
const $chat = document.getElementById("chat");
const $emptyState = document.getElementById("empty-state");
const $input = document.getElementById("msg-input");
const $btnSend = document.getElementById("btn-send");
const $btnNew = document.getElementById("btn-new");
const $btnReengage = document.getElementById("btn-reengage");
const $btnReset = document.getElementById("btn-reset");
const $btnStop = document.getElementById("btn-stop");
const $inputBar = document.getElementById("input-bar");
const $charCounter = document.getElementById("char-counter");
const MSG_MAX_LENGTH = 450;

// Sidebar
const $sidebar = document.querySelector(".sidebar");
const $sidebarResizer = document.getElementById("sidebar-resizer");
const $analyticsSection = document.getElementById("analytics-section");
const $detailsToggle = document.getElementById("details-toggle");
const $detailsContent = document.getElementById("details-content");
const $convListSection = document.querySelector(".conv-list-section");
const $convToggle = document.getElementById("conv-toggle");
const $convCount = document.getElementById("conv-count");
const $convList = document.getElementById("conv-list");
const $convListEmpty = document.getElementById("conv-list-empty");

const SIDEBAR_STORAGE_KEY = "orbio.sidebarWidth";
const SIDEBAR_DEFAULT_WIDTH = 280;
const SIDEBAR_MIN_WIDTH = 240;
const SIDEBAR_MAX_WIDTH = 520;
const MAIN_MIN_WIDTH = 360;

// ---- Theme toggle ----
const THEME_STORAGE_KEY = "orbio.theme";
const $themeToggle = document.getElementById("theme-toggle");

/**
 * Apply a theme to the document and persist it in localStorage.
 * @param {"light"|"dark"} theme
 */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem(THEME_STORAGE_KEY, theme);
}

// Initialize theme from storage (default: light)
(function initTheme() {
  const saved = localStorage.getItem(THEME_STORAGE_KEY) || "light";
  applyTheme(saved);
})();

$themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(current === "dark" ? "light" : "dark");
});

let sidebarResizeState = null;

// ---- API helpers ----

/**
 * Send a request to the backend API and return parsed JSON.
 * @param {string} path - URL path (e.g. "/conversations").
 * @param {RequestInit} [options] - Additional fetch options.
 * @returns {Promise<any>} Parsed JSON response.
 * @throws {Error} On non-2xx HTTP status.
 */
async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

// ---- Render helpers ----

/** Scroll the chat container to the bottom using a double-rAF for layout settling. */
function scrollToBottom() {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      $chat.scrollTop = $chat.scrollHeight;
    });
  });
}

/**
 * Scroll a specific element into view, or fall back to scrolling to bottom.
 * @param {HTMLElement|null} element
 */
function scrollElementIntoView(element) {
  if (!element) {
    scrollToBottom();
    return;
  }

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      element.scrollIntoView({ block: "end", behavior: "auto" });
    });
  });
}

/**
 * Append a chat message bubble to the chat area and auto-scroll.
 * @param {"user"|"assistant"} role
 * @param {string} content - Message text.
 * @returns {HTMLDivElement} The created message element.
 */
function appendMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = content;
  $chat.appendChild(div);
  scrollToBottom();
  return div;
}

/**
 * Show a "typing…" animation bubble in the chat while waiting for a response.
 * @returns {HTMLDivElement} The indicator element (removed later by {@link removeTypingIndicator}).
 */
function showTypingIndicator() {
  const div = document.createElement("div");
  div.className = "message assistant typing";
  div.id = "typing-indicator";
  div.innerHTML = '<div class="dots"><span></span><span></span><span></span></div>';
  $chat.appendChild(div);
  scrollToBottom();
  return div;
}

/** Remove the typing indicator bubble from the chat, if present. */
function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

/**
 * Enable or disable the message input bar and action buttons.
 * @param {boolean} enabled
 */
function setInputEnabled(enabled) {
  // When voice mode is active, the text input stays disabled regardless
  const voiceActive = window.voiceMode?.enabled;
  $input.disabled = voiceActive ? true : !enabled;
  $btnSend.disabled = !enabled;
  $btnStop.disabled = !enabled;
  if (enabled && !voiceActive) $input.focus();
}

/**
 * Show or hide the entire input bar (hidden when conversation is terminal).
 * @param {boolean} visible
 */
function setInputBarVisible(visible) {
  $inputBar.hidden = !visible;
}

/** Sync UI controls (e.g. re-engage button) with current conversation state. */
function updateConvInfo() {
  $btnReengage.disabled = !state.conversationId || state.status !== "in_progress";
}

/** @returns {boolean} True if the current conversation is in a terminal state. */
function isTerminal() {
  return ["qualified", "disqualified", "needs_review", "abandoned", "withdrawn"].includes(state.status);
}

/** Display a status banner at the bottom of the chat for a completed conversation. */
function showTerminalBanner() {
  const statusLabels = {
    qualified: "Conversation complete — status: qualified",
    disqualified: "Conversation complete — status: disqualified",
    needs_review: "Conversation complete — status: needs review",
    abandoned: "Conversation complete — status: abandoned",
    withdrawn: "Conversation ended — candidate withdrew",
  };

  const banner = document.createElement("div");
  banner.className = `terminal-banner ${state.status}`;
  banner.textContent = statusLabels[state.status] || `Conversation ended: ${state.status}`;
  $chat.appendChild(banner);
  scrollToBottom();
}

/**
 * Create a collapsible result panel (used for summary and extracted data).
 * @param {string} title - Panel heading text.
 * @param {string} bodyHtml - Inner HTML for the panel body.
 * @returns {HTMLElement}
 */
function createResultPanel(title, bodyHtml) {
  const panel = document.createElement("section");
  panel.className = "result-panel collapsed";

  const header = document.createElement("div");
  header.className = "result-panel-title";

  const titleSpan = document.createElement("span");
  titleSpan.textContent = title;

  const chevron = document.createElement("span");
  chevron.className = "chevron";
  chevron.textContent = "\u25BC";

  header.appendChild(titleSpan);
  header.appendChild(chevron);

  header.addEventListener("click", () => {
    panel.classList.toggle("collapsed");
  });

  const body = document.createElement("div");
  body.className = "panel-body";
  body.innerHTML = bodyHtml;

  panel.appendChild(header);
  panel.appendChild(body);
  return panel;
}

/** Fetch and display summary + extracted data panels after a conversation ends. */
async function showConversationResults() {
  try {
    const data = await api(`/conversations/${state.conversationId}`);
    let lastPanel = null;

    if (data.summary) {
      const panel = createResultPanel("Conversation Summary", escapeHtml(data.summary));
      $chat.appendChild(panel);
      lastPanel = panel;
    }

    if (data.extracted_data && Object.keys(data.extracted_data).length > 0) {
      const panel = createResultPanel(
        "Extracted Data",
        `<pre>${escapeHtml(JSON.stringify(data.extracted_data, null, 2))}</pre>`,
      );
      $chat.appendChild(panel);
      lastPanel = panel;
    }

    scrollElementIntoView(lastPanel);
  } catch (e) {
    console.error("Failed to load conversation results:", e);
  }
}

/**
 * Escape a string for safe insertion into HTML.
 * @param {string} text
 * @returns {string} HTML-escaped text.
 */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Convert a snake_case key to Title Case for display.
 * @param {string} value
 * @returns {string}
 */
function humanizeKey(value) {
  return String(value)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

/**
 * Format a stat value for display, handling nulls and floats.
 * @param {*} value
 * @returns {string}
 */
function formatStatValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number" && !Number.isInteger(value)) {
    return value.toFixed(1);
  }
  return String(value);
}

/**
 * Render a single stat card HTML fragment for the analytics breakdown.
 * @param {string} label
 * @param {*} value
 * @param {string} [extraClasses]
 * @returns {string} HTML string.
 */
function renderDetailCard(label, value, extraClasses = "") {
  const classes = ["details-stat-card"];
  if (extraClasses) {
    classes.push(extraClasses);
  }

  return `
    <div class="${classes.join(" ")}">
      <div class="details-stat-value">${escapeHtml(formatStatValue(value))}</div>
      <div class="details-stat-label">${escapeHtml(label)}</div>
    </div>
  `;
}

/** Ordered stage keys matching the backend STAGE_ORDER. */
const STAGE_ORDER = [
  "greeting", "name", "license", "city", "availability",
  "schedule", "experience_years", "experience_platform",
  "start_date", "closing",
];

/**
 * Render an inline horizontal bar chart for drop-off by stage.
 * Only stages with at least one drop-off are shown, ordered by funnel position.
 * @param {object} dropoffMap - { stage_key: count }
 * @returns {string} HTML string.
 */
function renderDropoffChart(dropoffMap) {
  const entries = STAGE_ORDER
    .filter((s) => dropoffMap[s] > 0)
    .map((s) => [s, dropoffMap[s]]);

  if (entries.length === 0) {
    return `<div class="dropoff-empty">No drop-offs yet</div>`;
  }

  const maxCount = Math.max(...entries.map(([, c]) => c));

  const rows = entries.map(([stage, count]) => {
    const pct = Math.max((count / maxCount) * 100, 6); // min 6% so tiny bars are visible
    return `
      <div class="dropoff-row">
        <span class="dropoff-label">${escapeHtml(humanizeKey(stage))}</span>
        <div class="dropoff-bar-track">
          <div class="dropoff-bar-fill" style="width:${pct.toFixed(1)}%"></div>
        </div>
        <span class="dropoff-count">${count}</span>
      </div>`;
  });

  return `<div class="dropoff-chart">${rows.join("")}</div>`;
}

/**
 * Render the full analytics breakdown (statuses, averages).
 * @param {object} data - Analytics response from the backend.
 * @returns {string} HTML string.
 */
function renderAnalyticsBreakdown(data) {
  const statusEntries = Object.entries(data.by_status || {});
  const statusCards = statusEntries.length > 0
    ? statusEntries.map(([status, count]) => renderDetailCard(humanizeKey(status), count))
    : [renderDetailCard("No statuses yet", "-", "wide muted")];

  const sentimentDisplay = data.avg_sentiment != null
    ? `${(data.avg_sentiment * 100).toFixed(0)}%`
    : "N/A";

  const averageCards = [
    renderDetailCard("Avg turns (qualified)", data.avg_turns_qualified),
    renderDetailCard("Avg turns (disqualified)", data.avg_turns_disqualified),
    renderDetailCard("Avg turns (needs review)", data.avg_turns_needs_review),
    renderDetailCard("Avg sentiment", sentimentDisplay),
  ];

  const dropoffBarChart = renderDropoffChart(data.dropoff_by_stage || {});

  return `
    <div class="details-stack">
      <section class="details-group">
        <div class="details-group-title">Statuses</div>
        <div class="details-grid">${statusCards.join("")}</div>
      </section>
      <section class="details-group">
        <div class="details-group-title">Average Turns</div>
        <div class="details-grid">${averageCards.join("")}</div>
      </section>
      <section class="details-group">
        <div class="details-group-title">Drop-off by Stage</div>
        ${dropoffBarChart}
      </section>
    </div>
  `;
}

// ---- Relative time formatting ----

/**
 * Format an ISO timestamp as a human-readable relative time string.
 * @param {string} isoString - ISO 8601 timestamp.
 * @returns {string} e.g. "just now", "5m ago", "2d ago", or a formatted date.
 */
function formatRelativeTime(isoString) {
  if (!isoString) return "";
  const now = Date.now();
  const then = new Date(isoString).getTime();
  const diffSec = Math.floor((now - then) / 1000);

  if (diffSec < 60) return "just now";
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  return new Date(isoString).toLocaleDateString();
}

// ---- Sidebar resizing ----

/** @returns {number} Maximum sidebar width based on current viewport. */
function getSidebarMaxWidth() {
  const viewportMax = window.innerWidth - MAIN_MIN_WIDTH - ($sidebarResizer?.offsetWidth || 0);
  return Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, viewportMax));
}

/**
 * Clamp a width value within the allowed sidebar range.
 * @param {number} width
 * @returns {number}
 */
function clampSidebarWidth(width) {
  return Math.min(getSidebarMaxWidth(), Math.max(SIDEBAR_MIN_WIDTH, Math.round(width)));
}

/**
 * Update ARIA min/max/current attributes on the sidebar resize handle.
 * @param {number} width - Current sidebar width in pixels.
 */
function updateSidebarResizeAria(width) {
  if (!$sidebarResizer) return;
  $sidebarResizer.setAttribute("aria-valuemin", String(SIDEBAR_MIN_WIDTH));
  $sidebarResizer.setAttribute("aria-valuemax", String(getSidebarMaxWidth()));
  $sidebarResizer.setAttribute("aria-valuenow", String(width));
}

/**
 * Apply a sidebar width to the CSS custom property and update ARIA attributes.
 * @param {number} width
 * @returns {number} The clamped width that was actually applied.
 */
function applySidebarWidth(width) {
  const nextWidth = clampSidebarWidth(width);
  document.documentElement.style.setProperty("--sidebar-width", `${nextWidth}px`);
  updateSidebarResizeAria(nextWidth);
  return nextWidth;
}

/**
 * Read the current rendered sidebar width from the DOM.
 * @returns {number} Width in pixels, or the default if unavailable.
 */
function getCurrentSidebarWidth() {
  return $sidebar?.getBoundingClientRect().width || SIDEBAR_DEFAULT_WIDTH;
}

/**
 * Save the sidebar width to localStorage for persistence across sessions.
 * @param {number} width - Width in pixels.
 */
function persistSidebarWidth(width) {
  try {
    window.localStorage.setItem(SIDEBAR_STORAGE_KEY, String(width));
  } catch (e) {
    console.warn("Could not persist sidebar width:", e);
  }
}

/** Restore sidebar width from localStorage (skipped on narrow viewports). */
function restoreSidebarWidth() {
  if (!$sidebar || window.innerWidth <= 768) {
    return;
  }

  let width = SIDEBAR_DEFAULT_WIDTH;
  try {
    const stored = Number.parseInt(window.localStorage.getItem(SIDEBAR_STORAGE_KEY) || "", 10);
    if (Number.isFinite(stored)) {
      width = stored;
    }
  } catch (e) {
    console.warn("Could not restore sidebar width:", e);
  }

  applySidebarWidth(width);
}

/** Re-clamp the sidebar width after a browser resize event. */
function syncSidebarWidthToViewport() {
  if (!$sidebar || window.innerWidth <= 768) {
    return;
  }

  applySidebarWidth(getCurrentSidebarWidth());
}

/** Finish a pointer-drag resize: persist the final width and clean up. */
function endSidebarResize() {
  if (!sidebarResizeState) {
    return;
  }

  const finalWidth = applySidebarWidth(getCurrentSidebarWidth());
  persistSidebarWidth(finalWidth);
  sidebarResizeState = null;
  document.body.classList.remove("sidebar-resizing");
}

/**
 * Begin a pointer-drag sidebar resize.
 * @param {number} clientX - Starting pointer X coordinate.
 */
function startSidebarResize(clientX) {
  if (!$sidebar || !$sidebarResizer || window.innerWidth <= 768) {
    return;
  }

  sidebarResizeState = {
    startX: clientX,
    startWidth: getCurrentSidebarWidth(),
  };
  document.body.classList.add("sidebar-resizing");
}

/**
 * Update sidebar width during an active pointer-drag resize.
 * @param {number} clientX - Current pointer X coordinate.
 */
function updateSidebarResize(clientX) {
  if (!sidebarResizeState) {
    return;
  }

  const delta = clientX - sidebarResizeState.startX;
  applySidebarWidth(sidebarResizeState.startWidth + delta);
}

// ---- Conversation list ----

/** Fetch all conversations from the backend and render the sidebar list. */
async function loadConversationList() {
  try {
    const items = await api("/conversations");
    if ($convCount) {
      $convCount.textContent = String(items.length);
    }
    // Clear list (keep empty placeholder)
    $convList.querySelectorAll(".conv-list-item").forEach((el) => el.remove());

    if (items.length === 0) {
      $convListEmpty.style.display = "block";
      return;
    }

    $convListEmpty.style.display = "none";

    for (const item of items) {
      const div = document.createElement("div");
      div.className = "conv-list-item";
      if (item.id === state.conversationId) {
        div.classList.add("active");
      }
      div.dataset.id = item.id;
      div.dataset.status = item.status;
      div.innerHTML = `
        <div class="conv-item-name">${escapeHtml(item.display_name)}</div>
        <div class="conv-item-meta">
          <span class="status-badge status-${item.status}">${item.status}</span>
          <span class="conv-item-time">${formatRelativeTime(item.last_activity)}</span>
        </div>
      `;
      div.addEventListener("click", () => selectConversation(item.id, item.status));
      $convList.appendChild(div);
    }
  } catch (e) {
    console.warn("Failed to load conversation list:", e.message);
  }
}

/**
 * Load and display an existing conversation from the sidebar list.
 * @param {string} id - Conversation UUID.
 * @param {string} status - Current conversation status.
 */
async function selectConversation(id, status) {
  // Cancel any in-flight fetch from a previous selectConversation call
  if (_selectConvController) {
    _selectConvController.abort();
  }
  _selectConvController = new AbortController();
  const { signal } = _selectConvController;

  // Update active state in list
  $convList.querySelectorAll(".conv-list-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === id);
  });

  state.conversationId = id;
  state.status = status;

  // Sync URL hash
  if (window.location.hash !== `#${id}`) {
    history.pushState(null, "", `#${id}`);
  }

  // Fetch conversation details and messages in parallel
  try {
    const [conv, messages] = await Promise.all([
      api(`/conversations/${id}`, { signal }),
      api(`/conversations/${id}/messages`, { signal }),
    ]);

    // If the user switched to a different conversation while we were fetching, bail out
    if (signal.aborted || state.conversationId !== id) return;

    state.status = conv.status;
    state.stage = conv.stage;
    state.messages = messages.map((m) => ({ role: m.role, content: m.content }));

    // Render chat
    $chat.innerHTML = "";
    for (const msg of state.messages) {
      appendMessage(msg.role, msg.content);
    }

    // Handle terminal vs in-progress
    if (isTerminal()) {
      stopMessagePolling();
      setInputBarVisible(false);
      setInputEnabled(false);
      showTerminalBanner();
      await showConversationResults();
    } else {
      setInputBarVisible(true);
      setInputEnabled(true);
      startMessagePolling();
    }

    updateConvInfo();
  } catch (e) {
    if (e.name === "AbortError") return; // user switched away — ignore
    if (handleStaleConversation(e)) return;
    appendSystemError(`Failed to load conversation: ${e.message}`);
  }
}

// ---- Analytics ----

/** Fetch analytics from the backend and update the sidebar stats. */
async function loadAnalytics() {
  try {
    const data = await api("/analytics");
    $analyticsSection.style.display = "block";
    document.getElementById("a-total").textContent = data.total_conversations;
    document.getElementById("a-qualified").textContent = data.by_status.qualified || 0;
    document.getElementById("a-completion").textContent = `${(data.completion_rate * 100).toFixed(0)}%`;
    document.getElementById("a-qualification").textContent = `${(data.qualification_rate * 100).toFixed(0)}%`;
    $detailsContent.innerHTML = renderAnalyticsBreakdown(data);
  } catch (e) {
    // Analytics are non-critical
    console.warn("Analytics unavailable:", e.message);
  }
}

// ---- Core actions ----

/** Create a new screening conversation, render the greeting, and update the sidebar. */
async function createConversation() {
  $btnNew.disabled = true;
  try {
    const data = await api("/conversations", { method: "POST" });
    state.conversationId = data.conversation_id;
    state.status = "in_progress";
    state.stage = "name";
    state.messages = [{ role: "assistant", content: data.greeting_message }];

    // Sync URL hash
    history.pushState(null, "", `#${data.conversation_id}`);

    // Clear chat and render greeting
    $chat.innerHTML = "";
    $emptyState?.remove();
    appendMessage("assistant", data.greeting_message);

    // Speak greeting in voice mode
    if (window.voiceMode?.enabled) {
      window.voiceMode.resumeAutoListen();
      window.voiceMode.onAssistantMessage(data.greeting_message, "es");
    }

    setInputBarVisible(true);
    setInputEnabled(true);
    startMessagePolling();
    updateConvInfo();

    // Refresh list and analytics in background
    loadConversationList();
    loadAnalytics();
  } catch (e) {
    appendSystemError("Cannot connect to backend. Is the FastAPI server running?");
  } finally {
    $btnNew.disabled = false;
  }
}

/** Send the current input text to the backend and display the agent's reply. */
async function sendMessage() {
  const text = $input.value.trim();
  if (!text || !state.conversationId) return;

  // Immediately show user message & clear input
  $input.value = "";
  resetTextareaHeight();
  $charCounter.classList.remove("visible", "warn", "limit");
  appendMessage("user", text);
  state.messages.push({ role: "user", content: text });

  // Disable input, show typing
  setInputEnabled(false);
  showTypingIndicator();

  try {
    // Include Whisper's detected language if available (from voice mode)
    const payload = { message: text };
    if (window.voiceMode?.lastWhisperLanguage) {
      payload.whisper_language = window.voiceMode.lastWhisperLanguage;
      window.voiceMode.lastWhisperLanguage = null; // consume it
    }

    const data = await api(`/conversations/${state.conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify(payload),
    });

    removeTypingIndicator();

    // Show assistant response
    appendMessage("assistant", data.response);
    state.messages.push({ role: "assistant", content: data.response });
    state.status = data.status;
    state.stage = data.stage;

    // User language (for STT) comes from the backend's language detection
    const userLang = data.language || "es";
    // TTS voice should match the response text language, not the user's language
    const ttsLang = /[áéíóúñ¿¡]/.test(data.response) ? "es" : "en";

    updateConvInfo();

    if (isTerminal()) {
      setInputBarVisible(false);
      setInputEnabled(false);
      showTerminalBanner();
      if (window.voiceMode?.enabled) {
        window.voiceMode.stopForTerminal();
        window.voiceMode.onAssistantMessage(data.response, userLang, ttsLang);
      }
      await showConversationResults();
      loadAnalytics();
    } else {
      setInputBarVisible(true);
      setInputEnabled(true);
      if (window.voiceMode?.enabled) {
        window.voiceMode.onAssistantMessage(data.response, userLang, ttsLang);
      }
    }

    // Refresh list so status/timestamp stay current
    loadConversationList();
  } catch (e) {
    removeTypingIndicator();
    if (handleStaleConversation(e)) return;
    appendSystemError(`Error: ${e.message}`);
    setInputEnabled(true);
  }
}

/** Manually trigger a re-engagement message for the current conversation. */
async function triggerReengagement() {
  if (!state.conversationId || state.status !== "in_progress") return;

  $btnReengage.disabled = true;
  try {
    await api(`/conversations/${state.conversationId}/reengage`, { method: "POST" });
    // Reload all messages to pick up re-engagement message
    await refreshMessages();
  } catch (e) {
    if (handleStaleConversation(e)) return;
    appendSystemError("Failed to trigger re-engagement");
  } finally {
    $btnReengage.disabled = state.status !== "in_progress";
  }
}

/** Re-fetch and re-render all messages for the active conversation. */
async function refreshMessages() {
  if (!state.conversationId) return;
  try {
    const messages = await api(`/conversations/${state.conversationId}/messages`);
    state.messages = messages.map((m) => ({ role: m.role, content: m.content }));

    // Re-render chat
    $chat.innerHTML = "";
    for (const msg of state.messages) {
      appendMessage(msg.role, msg.content);
    }
  } catch (e) {
    if (handleStaleConversation(e)) return;
    console.error("Failed to refresh messages:", e);
  }
}

/**
 * Start polling for new messages (e.g. server-side re-engagement nudges).
 * Only polls while the current conversation is in_progress.
 */
function startMessagePolling() {
  stopMessagePolling();
  _pollIntervalId = setInterval(async () => {
    if (!state.conversationId || isTerminal()) {
      stopMessagePolling();
      return;
    }
    try {
      const [conv, messages] = await Promise.all([
        api(`/conversations/${state.conversationId}`),
        api(`/conversations/${state.conversationId}/messages`),
      ]);

      // Detect status change (e.g. abandoned)
      if (conv.status !== state.status) {
        state.status = conv.status;
        state.stage = conv.stage;
      }

      // Re-render messages if count changed
      if (messages.length !== state.messages.length) {
        state.messages = messages.map((m) => ({ role: m.role, content: m.content }));
        $chat.innerHTML = "";
        for (const msg of state.messages) {
          appendMessage(msg.role, msg.content);
        }
      }

      // If conversation became terminal, update UI accordingly
      if (isTerminal()) {
        stopMessagePolling();
        setInputBarVisible(false);
        setInputEnabled(false);
        showTerminalBanner();
        await showConversationResults();
        updateConvInfo();
        loadConversationList();
        loadAnalytics();
      }
    } catch (e) {
      console.debug("Poll failed:", e);
    }
  }, POLL_INTERVAL_MS);
}

/** Stop the message polling interval. */
function stopMessagePolling() {
  if (_pollIntervalId) {
    clearInterval(_pollIntervalId);
    _pollIntervalId = null;
  }
}

/**
 * Display a red error message in the chat area.
 * @param {string} text
 */
function appendSystemError(text) {
  const div = document.createElement("div");
  div.className = "message assistant";
  div.style.background = "#ffeaea";
  div.style.color = "#c0392b";
  div.textContent = text;
  $chat.appendChild(div);
  scrollToBottom();
}

/** Delete all conversations and reload the page (after user confirmation). */
async function resetAllData() {
  if (!confirm("This will permanently delete ALL conversations and analytics. Are you sure?")) {
    return;
  }

  $btnReset.disabled = true;
  try {
    await api("/reset", { method: "DELETE" });
    // Clear hash and hard reload to get a clean slate
    history.replaceState(null, "", window.location.pathname);
    window.location.reload();
  } catch (e) {
    appendSystemError(`Reset failed: ${e.message}`);
    $btnReset.disabled = false;
  }
}

// ---- Event listeners ----

/** Send the [STOP] signal to immediately withdraw from the conversation. */
async function stopConversation() {
  if (!state.conversationId || isTerminal()) return;

  // Send the [STOP] signal as a message — backend handles it as immediate withdrawal
  $input.value = "[STOP]";
  await sendMessage();
}

$btnNew.addEventListener("click", createConversation);
$btnReengage.addEventListener("click", triggerReengagement);
$btnReset.addEventListener("click", resetAllData);
$btnSend.addEventListener("click", sendMessage);
$btnStop.addEventListener("click", stopConversation);

$input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

/** Reset the textarea height back to a single row. */
function resetTextareaHeight() {
  $input.style.height = "";
  $input.classList.remove("scrollable");
}

/** Auto-grow the textarea to fit content, up to its CSS max-height. */
function autoGrowTextarea() {
  // Collapse to single-row first so scrollHeight reflects actual content
  $input.style.height = "auto";
  const maxH = 120; // matches CSS max-height (3× the ~40px single row)
  const next = Math.min($input.scrollHeight, maxH);
  $input.style.height = `${next}px`;
  // Enable scrollbar only when content exceeds max height
  $input.classList.toggle("scrollable", $input.scrollHeight > maxH);
}

$input.addEventListener("input", () => {
  // Auto-grow
  autoGrowTextarea();

  // Char counter
  const len = $input.value.length;
  const remaining = MSG_MAX_LENGTH - len;

  if (len >= MSG_MAX_LENGTH * 0.75) {
    $charCounter.textContent = `${remaining}`;
    $charCounter.classList.add("visible");
    $charCounter.classList.toggle("warn", remaining <= 100 && remaining > 0);
    $charCounter.classList.toggle("limit", remaining <= 0);
  } else {
    $charCounter.classList.remove("visible", "warn", "limit");
  }
});

/**
 * Toggle the analytics details drawer open/closed.
 * @param {boolean} open
 */
function setAnalyticsDetailsOpen(open) {
  if (!$detailsToggle || !$detailsContent) return;
  $detailsContent.classList.toggle("open", open);
  $detailsToggle.classList.toggle("open", open);
  $detailsToggle.setAttribute("aria-expanded", String(open));
  $detailsToggle.setAttribute("aria-label", open ? "Hide analytics details" : "Show analytics details");
}

$detailsToggle?.addEventListener("click", () => {
  const open = !$detailsContent.classList.contains("open");
  setAnalyticsDetailsOpen(open);
});

/**
 * Collapse or expand the sidebar conversation list.
 * @param {boolean} collapsed
 */
function setConversationListCollapsed(collapsed) {
  if (!$convListSection || !$convToggle) return;
  $convListSection.classList.toggle("collapsed", collapsed);
  $convToggle.setAttribute("aria-expanded", String(!collapsed));
  $convToggle.setAttribute("aria-label", collapsed ? "Expand conversations" : "Collapse conversations");
}

$convToggle?.addEventListener("click", () => {
  const collapsed = !$convListSection.classList.contains("collapsed");
  setConversationListCollapsed(collapsed);
});

restoreSidebarWidth();

$sidebarResizer?.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) {
    return;
  }

  e.preventDefault();
  startSidebarResize(e.clientX);
});

window.addEventListener("pointermove", (e) => {
  updateSidebarResize(e.clientX);
});

window.addEventListener("pointerup", () => {
  endSidebarResize();
});

window.addEventListener("pointercancel", () => {
  endSidebarResize();
});

window.addEventListener("blur", () => {
  endSidebarResize();
});

window.addEventListener("resize", () => {
  syncSidebarWidthToViewport();
});

$sidebarResizer?.addEventListener("keydown", (e) => {
  if (window.innerWidth <= 768) {
    return;
  }

  const step = e.shiftKey ? 40 : 20;

  if (e.key === "ArrowLeft") {
    e.preventDefault();
    persistSidebarWidth(applySidebarWidth(getCurrentSidebarWidth() - step));
    return;
  }

  if (e.key === "ArrowRight") {
    e.preventDefault();
    persistSidebarWidth(applySidebarWidth(getCurrentSidebarWidth() + step));
    return;
  }

  if (e.key === "Home") {
    e.preventDefault();
    persistSidebarWidth(applySidebarWidth(SIDEBAR_MIN_WIDTH));
    return;
  }

  if (e.key === "End") {
    e.preventDefault();
    persistSidebarWidth(applySidebarWidth(getSidebarMaxWidth()));
  }
});

setAnalyticsDetailsOpen(false);
setConversationListCollapsed(true);

// ---- Stale conversation helper ----

/** Returns true if the error is a 404. If so, resets UI to the empty state. */
function handleStaleConversation(e) {
  if (!e.message?.includes("404")) return false;
  state.conversationId = null;
  state.status = null;
  state.messages = [];
  $chat.innerHTML = "";
  setInputBarVisible(false);
  history.replaceState(null, "", window.location.pathname);
  appendSystemError("This conversation no longer exists. It may have been deleted.");
  loadConversationList();
  return true;
}

// ---- Expose sendMessage for voice module ----
window.sendMessage = sendMessage;

// ---- URL hash routing ----

/** Restore conversation from the URL hash (e.g. #<uuid>) on load or navigation. */
async function loadConversationFromHash() {
  const id = window.location.hash.slice(1);
  if (!id || id === state.conversationId) return;
  // We don't know the status yet — selectConversation will fetch it
  await selectConversation(id, "unknown");
}

window.addEventListener("hashchange", loadConversationFromHash);

// ---- Init ----
loadConversationList();
loadAnalytics();
loadConversationFromHash();
