const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const filePreview = document.getElementById("file-preview");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const btnRemove = document.getElementById("btn-remove");
const btnAnalyze = document.getElementById("btn-analyze");
const btnReset = document.getElementById("btn-reset");

const processingStatus = document.getElementById("processing-status");
const progressFill = document.getElementById("progress-fill");

const verdictCard = document.getElementById("verdict-card");
const verdictIcon = document.getElementById("verdict-icon");
const verdictLabel = document.getElementById("verdict-label");
const verdictConfidence = document.getElementById("verdict-confidence");
const verdictMethod = document.getElementById("verdict-method");

const fakeProbabilityText = document.getElementById("fake-probability-text");
const fakeProbabilityBar = document.getElementById("fake-probability-bar");

const framesExtracted = document.getElementById("frames-extracted");
const facesDetected = document.getElementById("faces-detected");
const faceCropsUsed = document.getElementById("face-crops-used");
const clipsUsed = document.getElementById("clips-used");
const fusionGate = document.getElementById("fusion-gate");
const rppgRatio = document.getElementById("rppg-ratio");

const clipProbs = document.getElementById("clip-probs");
const rppgFeatures = document.getElementById("rppg-features");
const jsonOutput = document.getElementById("json-output");

const btnUserLogin = document.getElementById("btn-user-login");
const btnAdminLogin = document.getElementById("btn-admin-login");
const authModal = document.getElementById("auth-modal");
const authTitle = document.getElementById("auth-title");
const btnCloseModal = document.getElementById("btn-close-modal");

let selectedFile = null;
let progressInterval = null;


// =========================
// AUTH UI ONLY
// =========================
function openModal(title) {
    authTitle.textContent = title;
    authModal.classList.remove("hidden");
}

function closeModal() {
    authModal.classList.add("hidden");
}

btnUserLogin.addEventListener("click", () => openModal("User Login"));
btnAdminLogin.addEventListener("click", () => openModal("Admin Login"));
btnCloseModal.addEventListener("click", closeModal);
authModal.addEventListener("click", (e) => {
    if (e.target === authModal) closeModal();
});


// =========================
// FILE UPLOAD
// =========================
uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = "#3b82f6";
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.style.borderColor = "";
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = "";
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

btnRemove.addEventListener("click", () => {
    clearFile();
});

function handleFileSelect(file) {
    if (!file.type.startsWith("video/")) {
        alert("Please select a valid video file.");
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    filePreview.style.display = "flex";
    btnAnalyze.disabled = false;
}

function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    filePreview.style.display = "none";
    btnAnalyze.disabled = true;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}


// =========================
// PROCESSING STEPS
// =========================
function resetSteps() {
    ["step-extract", "step-face", "step-clips", "step-rppg", "step-fusion"].forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove("active", "done");
    });
    progressFill.style.width = "0%";
    processingStatus.textContent = "Waiting for upload...";
}

function startDemoProgress() {
    resetSteps();

    const steps = [
        { id: "step-extract", text: "Extracting frames from uploaded video...", pct: 20 },
        { id: "step-face", text: "Detecting and cropping face regions...", pct: 40 },
        { id: "step-clips", text: "Building temporal NeST clips...", pct: 60 },
        { id: "step-rppg", text: "Extracting rPPG signal features...", pct: 80 },
        { id: "step-fusion", text: "Running gated fusion and classifier...", pct: 96 }
    ];

    let idx = 0;
    progressInterval = setInterval(() => {
        if (idx > 0) {
            document.getElementById(steps[idx - 1].id).classList.remove("active");
            document.getElementById(steps[idx - 1].id).classList.add("done");
        }

        if (idx >= steps.length) {
            clearInterval(progressInterval);
            return;
        }

        document.getElementById(steps[idx].id).classList.add("active");
        processingStatus.textContent = steps[idx].text;
        progressFill.style.width = `${steps[idx].pct}%`;
        idx += 1;
    }, 900);
}


// =========================
// RESULT RENDERING
// =========================
function renderResult(data) {
    const isReal = data.label === "REAL";
    const confidencePct = (Number(data.confidence) * 100).toFixed(1);
    const fakePct = (Number(data.fake_probability) * 100).toFixed(1);

    verdictCard.className = `card verdict-card ${isReal ? "real" : "fake"}`;
    verdictIcon.textContent = isReal ? "✓" : "!";
    verdictLabel.textContent = isReal ? "AUTHENTIC" : "SYNTHETIC";
    verdictConfidence.textContent = `Confidence: ${confidencePct}%`;
    verdictMethod.textContent = `Method: ${data.fusion_method || "gated_visual_rppg_fusion"}`;

    fakeProbabilityText.textContent = `${fakePct}%`;
    fakeProbabilityBar.style.width = `${fakePct}%`;
    fakeProbabilityBar.className = `bar ${isReal ? "real" : "fake"}`;

    framesExtracted.textContent = data.frames_extracted ?? "--";
    facesDetected.textContent = data.faces_detected ?? "--";
    faceCropsUsed.textContent = data.face_crops_used ?? "--";
    clipsUsed.textContent = data.clips_used ?? "--";
    fusionGate.textContent = data.average_fusion_gate ?? "--";
    rppgRatio.textContent = data.rppg_windows_success_ratio ?? "--";

    clipProbs.innerHTML = "";
    if (Array.isArray(data.clip_fake_probabilities)) {
        data.clip_fake_probabilities.forEach(val => {
            const chip = document.createElement("span");
            chip.className = "chip";
            chip.textContent = val;
            clipProbs.appendChild(chip);
        });
    } else {
        clipProbs.textContent = "--";
    }

    rppgFeatures.innerHTML = "";
    if (data.rppg_features) {
        Object.entries(data.rppg_features).forEach(([key, value]) => {
            const div = document.createElement("div");
            div.className = "feature-mini";
            div.innerHTML = `
                <span class="mini-key">${key}</span>
                <span class="mini-value">${value}</span>
            `;
            rppgFeatures.appendChild(div);
        });
    }

    jsonOutput.textContent = JSON.stringify(data, null, 2);
}


// =========================
// RESET
// =========================
btnReset.addEventListener("click", () => {
    clearFile();
    resetSteps();

    verdictCard.className = "card verdict-card neutral";
    verdictIcon.textContent = "—";
    verdictLabel.textContent = "NO RESULT";
    verdictConfidence.textContent = "Confidence: --";
    verdictMethod.textContent = "Method: gated_visual_rppg_fusion";

    fakeProbabilityText.textContent = "--";
    fakeProbabilityBar.style.width = "0%";

    framesExtracted.textContent = "--";
    facesDetected.textContent = "--";
    faceCropsUsed.textContent = "--";
    clipsUsed.textContent = "--";
    fusionGate.textContent = "--";
    rppgRatio.textContent = "--";

    clipProbs.innerHTML = "--";
    rppgFeatures.innerHTML = "";
    jsonOutput.textContent = "No output yet.";
});


// =========================
// DETECTION REQUEST
// =========================
btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("video", selectedFile);

    startDemoProgress();

    try {
        const response = await fetch("/detect/", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Detection failed.");
        }

        clearInterval(progressInterval);
        ["step-extract", "step-face", "step-clips", "step-rppg", "step-fusion"].forEach(id => {
            const el = document.getElementById(id);
            el.classList.remove("active");
            el.classList.add("done");
        });
        progressFill.style.width = "100%";
        processingStatus.textContent = "Prediction complete.";

        renderResult(data);
    } catch (err) {
        clearInterval(progressInterval);
        processingStatus.textContent = `Error: ${err.message}`;
        alert(err.message);
    }
});

resetSteps();
