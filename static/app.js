const API_BASE = "";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const stageWrap = document.getElementById("stageWrap");
const stagePlaceholder = document.getElementById("stagePlaceholder");
const exercisePill = document.getElementById("exercisePill");

const userIdInput = document.getElementById("userIdInput");
const exerciseSelect = document.getElementById("exerciseSelect");
const targetSecondsGroup = document.getElementById("targetSecondsGroup");
const targetSecondsInput = document.getElementById("targetSeconds");
const startBtn = document.getElementById("startBtn");
const endBtn = document.getElementById("endBtn");
const statusHint = document.getElementById("statusHint");

const repLabel = document.getElementById("repLabel");
const repValue = document.getElementById("repValue");
const formValue = document.getElementById("formValue");
const formBarFill = document.getElementById("formBarFill");
const feedbackList = document.getElementById("feedbackList");

const summaryCard = document.getElementById("summaryCard");
const summaryGrid = document.getElementById("summaryGrid");
const recommendBtn = document.getElementById("recommendBtn");
const recommendationBox = document.getElementById("recommendationBox");

const connDot = document.getElementById("connDot");
const connLabel = document.getElementById("connLabel");

let mediaStream = null;
let sessionId = null;
let rafHandle = null;
let captureInterval = null;
let lastLandmarks = null;
let lastRepCount = 0;
let currentUserId = null;
let currentExercise = null;

const SKELETON_CONNECTIONS = [
  ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
  ["LEFT_SHOULDER", "LEFT_ELBOW"], ["LEFT_ELBOW", "LEFT_WRIST"],
  ["RIGHT_SHOULDER", "RIGHT_ELBOW"], ["RIGHT_ELBOW", "RIGHT_WRIST"],
  ["LEFT_SHOULDER", "LEFT_HIP"], ["RIGHT_SHOULDER", "RIGHT_HIP"],
  ["LEFT_HIP", "RIGHT_HIP"],
  ["LEFT_HIP", "LEFT_KNEE"], ["LEFT_KNEE", "LEFT_ANKLE"],
  ["RIGHT_HIP", "RIGHT_KNEE"], ["RIGHT_KNEE", "RIGHT_ANKLE"],
];

// ---------------------------------------------------------------------
// Backend health check
// ---------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      connDot.classList.add("ok");
      connLabel.textContent = "Backend connected";
    } else {
      throw new Error("bad status");
    }
  } catch (e) {
    connDot.classList.add("bad");
    connLabel.textContent = "Backend unreachable";
  }
}
checkHealth();

exerciseSelect.addEventListener("change", () => {
  targetSecondsGroup.style.display = exerciseSelect.value === "plank" ? "flex" : "none";
});

// ---------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------
startBtn.addEventListener("click", startSession);
endBtn.addEventListener("click", endSession);
recommendBtn.addEventListener("click", fetchRecommendation);

async function startSession() {
  const userId = userIdInput.value.trim();
  if (!userId) {
    statusHint.textContent = "Please enter a name/user id first.";
    return;
  }
  currentUserId = userId;
  currentExercise = exerciseSelect.value;

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  } catch (e) {
    statusHint.textContent = "Camera access was denied or unavailable. Please allow camera permission and try again.";
    return;
  }

  video.srcObject = mediaStream;
  await video.play();

  const res = await fetch(`${API_BASE}/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, exercise: currentExercise }),
  });
  if (!res.ok) {
    statusHint.textContent = "Could not start a session — is the backend running?";
    stopStream();
    return;
  }
  const data = await res.json();
  sessionId = data.session_id;

  stagePlaceholder.classList.add("hidden");
  summaryCard.style.display = "none";
  recommendationBox.style.display = "none";
  resetLiveStats();

  startBtn.disabled = true;
  endBtn.disabled = false;
  userIdInput.disabled = true;
  exerciseSelect.disabled = true;
  statusHint.textContent = "Stand back so your full body is visible, then perform the exercise.";

  renderLoop();
  captureInterval = setInterval(captureAndAnalyze, 350);
}

async function endSession() {
  clearInterval(captureInterval);
  captureInterval = null;

  const targetSeconds = parseFloat(targetSecondsInput.value) || 60;
  let summary = null;
  if (sessionId) {
    try {
      const res = await fetch(`${API_BASE}/session/${sessionId}/end`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_seconds: targetSeconds }),
      });
      if (res.ok) {
        const data = await res.json();
        summary = data.summary;
      }
    } catch (e) {
      // backend may already be down; still tear down the local UI
    }
  }

  stopStream();
  sessionId = null;
  startBtn.disabled = false;
  endBtn.disabled = true;
  userIdInput.disabled = false;
  exerciseSelect.disabled = false;
  stagePlaceholder.classList.remove("hidden");
  exercisePill.textContent = "not started";
  stageWrap.classList.remove("form-good", "form-warn", "form-bad");
  statusHint.textContent = "Session ended. Fill in your details and press start to begin a new one.";

  if (summary) {
    renderSummary(summary);
  }
}

function stopStream() {
  if (rafHandle) cancelAnimationFrame(rafHandle);
  rafHandle = null;
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  lastLandmarks = null;
}

function resetLiveStats() {
  lastRepCount = 0;
  repValue.textContent = "0";
  formValue.textContent = "\u2014";
  formBarFill.style.width = "0%";
  formBarFill.style.background = "var(--text-faint)";
  feedbackList.innerHTML = '<li class="feedback-empty">Coaching messages will appear here once form issues are detected.</li>';
}

// ---------------------------------------------------------------------
// Render loop: draw mirrored video frame + skeleton overlay every frame
// ---------------------------------------------------------------------
function renderLoop() {
  if (!mediaStream) return;
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  if (lastLandmarks) {
    drawSkeleton(lastLandmarks);
  }

  rafHandle = requestAnimationFrame(renderLoop);
}

function drawSkeleton(landmarks) {
  const w = canvas.width, h = canvas.height;
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(200,255,77,0.85)";
  ctx.fillStyle = "rgba(200,255,77,0.95)";

  for (const [a, b] of SKELETON_CONNECTIONS) {
    const pa = landmarks[a], pb = landmarks[b];
    if (!pa || !pb) continue;
    ctx.beginPath();
    ctx.moveTo(pa[0] * w, pa[1] * h);
    ctx.lineTo(pb[0] * w, pb[1] * h);
    ctx.stroke();
  }
  for (const name in landmarks) {
    const [x, y] = landmarks[name];
    ctx.beginPath();
    ctx.arc(x * w, y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---------------------------------------------------------------------
// Frame capture + analysis
// ---------------------------------------------------------------------
async function captureAndAnalyze() {
  if (!sessionId) return;
  const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
  const base64 = dataUrl.split(",")[1];

  try {
    const res = await fetch(`${API_BASE}/session/${sessionId}/analyze_frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: base64 }),
    });
    if (!res.ok) return;
    const data = await res.json();
    handleAnalysisResult(data);
  } catch (e) {
    // transient network hiccup; next tick will retry
  }
}

function handleAnalysisResult(data) {
  if (!data.person_detected) {
    exercisePill.textContent = "move into frame";
    lastLandmarks = null;
    return;
  }

  lastLandmarks = data.landmarks || null;

  if (!data.active_exercise) {
    exercisePill.textContent = "detecting exercise\u2026";
    return;
  }

  exercisePill.textContent = data.active_exercise.replace("_", " ");

  if (typeof data.rep_count === "number") {
    repLabel.textContent = "Reps";
    if (data.rep_count !== lastRepCount) {
      repValue.textContent = data.rep_count;
      repValue.classList.remove("tick");
      void repValue.offsetWidth; // restart animation
      repValue.classList.add("tick");
      lastRepCount = data.rep_count;
    }
  } else if (typeof data.hold_seconds === "number") {
    repLabel.textContent = "Hold (s)";
    repValue.textContent = data.hold_seconds.toFixed(1);
  }

  if (typeof data.form_score === "number") {
    formValue.textContent = Math.round(data.form_score) + "%";
    formBarFill.style.width = Math.max(0, Math.min(100, data.form_score)) + "%";

    let colorClass = "form-bad";
    let barColor = "var(--coral)";
    if (data.form_score >= 85) { colorClass = "form-good"; barColor = "var(--lime)"; }
    else if (data.form_score >= 60) { colorClass = "form-warn"; barColor = "var(--amber)"; }

    stageWrap.classList.remove("form-good", "form-warn", "form-bad");
    stageWrap.classList.add(colorClass);
    formBarFill.style.background = barColor;
  }

  if (data.feedback && data.feedback.length > 0) {
    for (const msg of data.feedback) {
      addFeedbackMessage(msg, data.issues);
    }
  }
}

function addFeedbackMessage(text, issues) {
  let severity = "positive";
  if (issues && issues.length) {
    const match = issues.find((i) => i.message === text);
    if (match) severity = match.severity === "major" ? "major" : "minor";
  }

  const emptyPlaceholder = feedbackList.querySelector(".feedback-empty");
  if (emptyPlaceholder) emptyPlaceholder.remove();

  const li = document.createElement("li");
  li.textContent = text;
  li.classList.add(`severity-${severity}`);
  feedbackList.prepend(li);

  while (feedbackList.children.length > 8) {
    feedbackList.removeChild(feedbackList.lastChild);
  }
}

// ---------------------------------------------------------------------
// Summary + recommendation
// ---------------------------------------------------------------------
function renderSummary(summary) {
  summaryCard.style.display = "block";
  recommendationBox.style.display = "none";

  const rows = [
    ["Form accuracy", summary.form_accuracy, "%"],
    ["Rep accuracy", summary.rep_accuracy, "%"],
    ["Range of motion", summary.rom_pct, "%"],
    ["Consistency", summary.consistency, "%"],
    ["Completion", summary.completion_pct, "%"],
    ["Total reps", summary.total_reps, ""],
  ];

  summaryGrid.innerHTML = rows.map(([label, value, suffix]) => `
    <div class="summary-item">
      <span class="summary-item-label">${label}</span>
      <span class="summary-item-value">${value !== null && value !== undefined ? Math.round(value) : "\u2014"}${suffix}</span>
    </div>
  `).join("");
}

async function fetchRecommendation() {
  if (!currentUserId || !currentExercise || currentExercise === "auto") {
    recommendationBox.style.display = "block";
    recommendationBox.innerHTML = "<p>Recommendations are available for a specific exercise — start a session with a fixed exercise (not auto-detect) to get one.</p>";
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/users/${encodeURIComponent(currentUserId)}/recommendation?exercise=${encodeURIComponent(currentExercise)}`);
    if (!res.ok) throw new Error("request failed");
    const rec = await res.json();
    recommendationBox.style.display = "block";
    recommendationBox.innerHTML = `
      <h3>${rec.difficulty} &middot; ${rec.target_sets} sets &times; ${rec.target_reps} reps</h3>
      <p>${rec.rationale}</p>
      <ul>${rec.focus_cues.map((c) => `<li>${c}</li>`).join("")}</ul>
    `;
  } catch (e) {
    recommendationBox.style.display = "block";
    recommendationBox.innerHTML = "<p>Could not fetch a recommendation right now.</p>";
  }
}
