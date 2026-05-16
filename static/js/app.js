/**
 * SHL Assessment Recommender — chat frontend
 */

const MAX_TURNS = 8;

const TEST_TYPE_LABELS = {
  A: "Ability",
  P: "Personality",
  B: "Biodata",
  K: "Knowledge",
  S: "Situational Judgment",
  C: "Competency",
};

const messagesEl = document.getElementById("messages");
const welcomeEl = document.getElementById("welcome");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const btnSend = document.getElementById("btnSend");
const btnNewChat = document.getElementById("btnNewChat");
const turnBadge = document.getElementById("turnBadge");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

/** @type {{ role: 'user' | 'assistant', content: string }[]} */
let conversation = [];
let isLoading = false;
let conversationEnded = false;

function updateTurnBadge() {
  const userTurns = conversation.filter((m) => m.role === "user").length;
  turnBadge.textContent = `Turn ${userTurns} / ${MAX_TURNS}`;
}

function setLoading(loading) {
  isLoading = loading;
  btnSend.disabled = loading || conversationEnded;
  userInput.disabled = conversationEnded;
  if (!conversationEnded) {
    userInput.disabled = loading;
  }
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    if (data.status === "ok") {
      statusDot.className = "status-dot ok";
      statusText.textContent = "Server online";
      return true;
    }
    throw new Error("Unexpected health response");
  } catch {
    statusDot.className = "status-dot error";
    statusText.textContent = "Server offline";
    return false;
  }
}

function hideWelcome() {
  if (welcomeEl) welcomeEl.style.display = "none";
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function createTypingIndicator() {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";
  wrap.id = "typingIndicator";
  wrap.innerHTML = `
    <div class="typing" aria-label="Assistant is typing">
      <span></span><span></span><span></span>
    </div>
  `;
  messagesEl.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

function renderRecommendations(recs) {
  if (!recs || recs.length === 0) return "";

  const cards = recs
    .map((rec) => {
      const type = (rec.test_type || "K").toUpperCase().charAt(0);
      const label = TEST_TYPE_LABELS[type] || type;
      const safeName = escapeHtml(rec.name);
      const safeUrl = escapeHtml(rec.url);
      return `
        <a class="rec-card" href="${safeUrl}" target="_blank" rel="noopener noreferrer">
          <h4>${safeName}</h4>
          <div class="rec-meta">
            <span class="type-badge">${escapeHtml(type)} · ${escapeHtml(label)}</span>
          </div>
          <div class="rec-link">View on SHL.com →</div>
        </a>
      `;
    })
    .join("");

  return `<div class="recommendations">${cards}</div>`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function appendMessage(role, content, recommendations = []) {
  hideWelcome();

  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  wrap.appendChild(bubble);

  if (role === "assistant" && recommendations.length > 0) {
    const recHtml = renderRecommendations(recommendations);
    if (recHtml) {
      const recContainer = document.createElement("div");
      recContainer.innerHTML = recHtml;
      wrap.appendChild(recContainer.firstElementChild);
    }
  }

  messagesEl.appendChild(wrap);
  scrollToBottom();
}

function showError(text) {
  hideWelcome();
  const el = document.createElement("div");
  el.className = "error-banner";
  el.textContent = text;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function showEndBanner() {
  const el = document.createElement("div");
  el.className = "end-banner";
  el.textContent = "Conversation complete. Start a new chat to continue.";
  messagesEl.appendChild(el);
  scrollToBottom();
}

async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed || isLoading || conversationEnded) return;

  const userTurns = conversation.filter((m) => m.role === "user").length;
  if (userTurns >= MAX_TURNS) {
    showError(`Maximum ${MAX_TURNS} turns reached. Please start a new conversation.`);
    return;
  }

  conversation.push({ role: "user", content: trimmed });
  appendMessage("user", trimmed);
  updateTurnBadge();

  userInput.value = "";
  userInput.style.height = "auto";
  setLoading(true);
  createTypingIndicator();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: conversation }),
    });

    removeTypingIndicator();

    let data;
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      data = await res.json();
    } else {
      throw new Error("Invalid response from server");
    }

    if (!res.ok) {
      const msg = data.reply || "Something went wrong. Please try again.";
      showError(msg);
      conversation.pop();
      updateTurnBadge();
      return;
    }

    const reply = data.reply || "";
    const recommendations = Array.isArray(data.recommendations)
      ? data.recommendations
      : [];

    conversation.push({ role: "assistant", content: reply });
    appendMessage("assistant", reply, recommendations);
    updateTurnBadge();

    if (data.end_of_conversation) {
      conversationEnded = true;
      userInput.placeholder = "Conversation ended";
      setLoading(false);
      showEndBanner();
    }
  } catch (err) {
    removeTypingIndicator();
    console.error(err);
    showError("Could not reach the server. Check that it is running and try again.");
    conversation.pop();
    updateTurnBadge();
  } finally {
    if (!conversationEnded) setLoading(false);
    userInput.focus();
  }
}

function resetChat() {
  conversation = [];
  conversationEnded = false;
  messagesEl.innerHTML = "";
  if (welcomeEl) {
    welcomeEl.style.display = "";
    messagesEl.appendChild(welcomeEl);
  }
  userInput.value = "";
  userInput.placeholder = "Describe your hiring need…";
  userInput.disabled = false;
  updateTurnBadge();
  setLoading(false);
  userInput.focus();
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(userInput.value);
});

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = `${Math.min(userInput.scrollHeight, 160)}px`;
});

btnNewChat.addEventListener("click", resetChat);

document.querySelectorAll(".hint-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const prompt = chip.getAttribute("data-prompt");
    if (prompt) {
      userInput.value = prompt;
      userInput.dispatchEvent(new Event("input"));
      chatForm.requestSubmit();
    }
  });
});

updateTurnBadge();
checkHealth();
setInterval(checkHealth, 30000);
userInput.focus();
