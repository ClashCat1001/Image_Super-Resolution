const panelForm = document.querySelector(".panel-form");
const fileInput = document.getElementById("files");
const uploadBox = document.querySelector(".upload-box");
const urlModeInfo = document.getElementById("url-mode-info");
const urlInput = document.getElementById("image-urls");
const fileCount = document.getElementById("file-count");
const selectedFiles = document.getElementById("selected-files");
const statusBox = document.getElementById("status-box");
const results = document.getElementById("results");
const submitButton = document.getElementById("submit-button");
const denoiseInput = document.getElementById("denoise-level");
const denoiseValue = document.getElementById("denoise-value");
const processingModeSelect = document.getElementById("processing-mode");
const processingHint = document.getElementById("processing-hint");
const deviceModeSelect = document.getElementById("device-mode");
const runtimeHint = document.getElementById("runtime-hint");
const runtimeDetail = document.getElementById("runtime-detail");
const form = document.getElementById("upscale-form");
const folderModeInfo = document.getElementById("folder-mode-info");
const refreshFolderButton = document.getElementById("refresh-folder");
const sourceModeInputs = Array.from(
  document.querySelectorAll('input[name="source_mode"]'),
);

let currentFolderFiles = [];
let runtimeInfo = null;
let inputFolderLoading = false;
let processing = false;

function formatBytes(bytes) {
  if (bytes === 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  const value = Math.floor(Math.log(bytes) / Math.log(1024));
  const size = bytes / 1024 ** value;
  return `${size.toFixed(size >= 10 || value === 0 ? 0 : 1)} ${units[value]}`;
}

function getSourceMode() {
  return sourceModeInputs.find((input) => input.checked)?.value || "upload";
}

function getImageUrls() {
  return urlInput.value.split(/\r\n|\r|\n/).map((line) => line.trim()).filter(Boolean);
}

function formatDenoiseLabel(level) {
  return `d${level}x`;
}

function updateRuntimeUI(payload) {
  runtimeInfo = payload;
  const available = payload.available_devices || {};
  const cudaAvailable = Boolean(available.cuda);
  const metalAvailable = Boolean(available.metal);
  const cudaOption = Array.from(deviceModeSelect.options).find(
    (option) => option.value === "cuda",
  );

  if (cudaOption) {
    cudaOption.disabled = !cudaAvailable;
    cudaOption.textContent = cudaAvailable ? "CUDA" : "CUDA（不可用）";
  }
  const metalOption = Array.from(deviceModeSelect.options).find(
    (option) => option.value === "metal",
  );
  if (metalOption) {
    metalOption.disabled = !metalAvailable;
    metalOption.textContent = metalAvailable ? "Metal" : "Metal（不可用）";
  }

  if (cudaAvailable) {
    runtimeHint.textContent = "CUDA 可用";
    runtimeDetail.textContent = "自动模式会优先尝试 CUDA，其次是 Metal，最后回退到 CPU。";
  } else if (metalAvailable) {
    runtimeHint.textContent = "Metal 可用";
    runtimeDetail.textContent = "自动模式会优先尝试 Metal，异常时回退到 CPU。";
  } else {
    runtimeHint.textContent = "当前走 CPU";
    runtimeDetail.textContent = `当前后端使用 ${payload.cpu_threads || "?"} 线程 CPU 推理。`;
  }

  if (!cudaAvailable && deviceModeSelect.value === "cuda") {
    deviceModeSelect.value = "auto";
  }

  if (!metalAvailable && deviceModeSelect.value === "metal") {
    deviceModeSelect.value = "auto";
  }
}

async function loadRuntimeInfo() {
  try {
    const response = await fetch("/api/runtime");
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "读取运行时信息失败。");
    }

    updateRuntimeUI(payload);
  } catch (error) {
    runtimeHint.textContent = "检测失败";
    runtimeDetail.textContent = error.message || "无法读取运行时信息。";
  }
}

function setStatus(kind, message) {
  statusBox.className = `status-box ${kind}`;
  statusBox.textContent = message;
}

function setFileList(entries, emptyMessage) {
  if (!entries.length) {
    selectedFiles.className = "file-list empty-state";
    selectedFiles.innerHTML = `<li>${emptyMessage}</li>`;
    return;
  }

  selectedFiles.className = "file-list";
  selectedFiles.innerHTML = "";

  entries.forEach((entry) => {
    const item = document.createElement("li");
    const name = document.createElement("span");
    const meta = document.createElement("span");

    name.textContent = entry.name;
    meta.className = "file-meta";
    meta.textContent = entry.meta;

    item.append(name, meta);
    selectedFiles.appendChild(item);
  });
}

function renderUploadFiles() {
  const files = Array.from(fileInput.files ?? []);
  fileCount.textContent = `${files.length} 张图片`;

  setFileList(
    files.map((file) => ({
      name: file.name,
      meta: formatBytes(file.size),
    })),
    "还没有选择图片。",
  );
}

function renderUrlFiles() {
  const urls = getImageUrls();
  fileCount.textContent = `${urls.length} 个链接`;
  setFileList(
    urls.map((url) => ({ name: url, meta: "待下载" })),
    "还没有输入下载 URL，请每行输入一个图片链接。",
  );
}

function renderInputFolderFiles(files) {
  currentFolderFiles = files;
  fileCount.textContent = `${files.length} 张图片`;

  setFileList(
    files.map((name) => ({
      name,
      meta: "Images_Input",
    })),
    "Images_Input 中还没有可处理的图片。",
  );
}

async function loadInputFolderFiles({ keepStatus = true } = {}) {
  inputFolderLoading = true;
  submitButton.disabled = true;
  try {
    const response = await fetch("/api/input-images");
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "读取 Images_Input 失败。");
    }

    currentFolderFiles = payload.files || [];
    if (getSourceMode() === "input-folder") {
      renderInputFolderFiles(currentFolderFiles);
    }
    if (!keepStatus && getSourceMode() === "input-folder") {
      setStatus(
        "idle",
        `已扫描 ${payload.count} 张待处理图片。处理时会跳过 Archive，并在成功后归档原图。`,
      );
    }
  } catch (error) {
    currentFolderFiles = [];
    if (getSourceMode() === "input-folder") {
      renderInputFolderFiles([]);
      setStatus("error", error.message || "读取 Images_Input 失败。");
    }
  } finally {
    inputFolderLoading = false;
    submitButton.disabled = processing;
  }
}

function renderResults(payload) {
  const cards = payload.results ?? [];
  const failures = payload.failures ?? [];

  if (!cards.length && !failures.length) {
    results.className = "results-grid empty-state";
    results.innerHTML = "<p>本次没有生成结果。</p>";
    return;
  }

  results.className = "results-grid";
  results.innerHTML = "";

  cards.forEach((item) => {
    const card = document.createElement("article");
    card.className = "result-card";

    const image = document.createElement("img");
    image.src = item.preview_url;
    image.alt = item.output_name;
    image.loading = "lazy";

    const body = document.createElement("div");
    body.className = "result-body";

    const title = document.createElement("div");
    title.className = "result-title";
    title.textContent = item.output_name;

    const input = document.createElement("div");
    input.className = "result-detail";
    input.textContent = `来源：${item.input_name}`;

    const size = document.createElement("div");
    size.className = "result-detail";
    size.textContent = `尺寸：${item.width} × ${item.height}`;

    const params = document.createElement("div");
    params.className = "result-detail";
    const modeLabel = item.processing_mode === "low" ? "清晰原图 · 低处理" : "标准超分";
    params.textContent = `倍率：${item.scale}x ｜ 降噪：${item.denoise_label} ｜ ${modeLabel}`;

    const path = document.createElement("div");
    path.className = "result-detail";
    path.textContent = item.output_path;

    body.append(title, input, size, params, path);

    if (item.archive_path) {
      const archive = document.createElement("div");
      archive.className = "result-detail";
      archive.textContent = `已归档：${item.archive_path}`;
      body.appendChild(archive);
    }

    card.append(image, body);
    results.appendChild(card);
  });

  if (failures.length) {
    const failureList = document.createElement("div");
    failureList.className = "failure-list";

    failures.forEach((item) => {
      const failure = document.createElement("div");
      failure.className = "failure-item";

      const title = document.createElement("strong");
      title.textContent = item.input_name;

      const message = document.createElement("div");
      message.textContent = item.error;

      failure.append(title, message);
      failureList.appendChild(failure);
    });

    results.appendChild(failureList);
  }
}

function syncModeUI({ refreshFolder = false } = {}) {
  const mode = getSourceMode();

  sourceModeInputs.forEach((input) => {
    const card = input.closest(".source-card");
    card?.classList.toggle("active", input.checked);
  });

  const folderMode = mode === "input-folder";
  const urlMode = mode === "url";
  panelForm.classList.toggle("folder-mode", folderMode);
  folderModeInfo.classList.toggle("hidden", !folderMode);
  urlModeInfo.classList.toggle("hidden", !urlMode);
  uploadBox.classList.toggle("hidden", urlMode);
  fileInput.disabled = mode !== "upload";
  urlInput.disabled = !urlMode;

  if (folderMode) {
    if (refreshFolder || !currentFolderFiles.length) {
      loadInputFolderFiles({ keepStatus: false });
    } else {
      renderInputFolderFiles(currentFolderFiles);
    }
  } else if (urlMode) {
    renderUrlFiles();
  } else {
    renderUploadFiles();
  }
}

fileInput.addEventListener("change", () => {
  if (getSourceMode() === "upload") {
    renderUploadFiles();
  }
});

urlInput.addEventListener("input", () => {
  if (getSourceMode() === "url") {
    renderUrlFiles();
  }
});

sourceModeInputs.forEach((input) => {
  input.addEventListener("change", () => syncModeUI({ refreshFolder: true }));
});

refreshFolderButton.addEventListener("click", () => {
  loadInputFolderFiles({ keepStatus: false });
});

function syncProcessingUI() {
  const low = processingModeSelect.value === "low";
  const requestedLevel = Number.parseInt(denoiseInput.value, 10);
  denoiseValue.textContent = formatDenoiseLabel(low ? Math.min(requestedLevel, 2) : requestedLevel);
  processingHint.textContent = low
    ? "适合清晰原图：75% Lanczos 放大 + 25% 模型结果，降噪最高 d2x，减少涂抹感。"
    : "标准模式按所选降噪等级处理。";
}

processingModeSelect.addEventListener("change", syncProcessingUI);
denoiseInput.addEventListener("input", syncProcessingUI);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (processing) {
    return;
  }

  const sourceMode = getSourceMode();
  const files = Array.from(fileInput.files ?? []);
  const urls = getImageUrls();

  if (sourceMode === "upload" && !files.length) {
    setStatus("error", "请先选择至少一张图片。");
    return;
  }

  if (sourceMode === "url" && (!urls.length || urls.length > 100)) {
    setStatus("error", urls.length ? "每批最多支持 100 个 URL，请分批处理。" : "请先输入图片下载 URL，每行一个链接。");
    return;
  }

  if (sourceMode === "input-folder" && !currentFolderFiles.length) {
    setStatus("error", "Images_Input 中还没有可处理的图片，请先放入图片后刷新。");
    return;
  }

  if (sourceMode === "input-folder" && inputFolderLoading) {
    setStatus("error", "目录扫描尚未完成，请稍后再试。");
    return;
  }

  const formData = new FormData();
  formData.append("source_mode", sourceMode);
  formData.append("device_mode", deviceModeSelect.value);
  formData.append("processing_mode", processingModeSelect.value);
  formData.append("scale", document.getElementById("scale").value);
  formData.append("denoise_level", denoiseInput.value);

  if (sourceMode === "upload") {
    files.forEach((file) => formData.append("files", file));
  } else if (sourceMode === "url") {
    formData.append("urls", urls.join("\n"));
  }

  processing = true;
  submitButton.disabled = true;
  const pendingCount = sourceMode === "url" ? urls.length : sourceMode === "upload" ? files.length : currentFolderFiles.length;
  setStatus("running", sourceMode === "url"
    ? `正在下载并超分 ${pendingCount} 个链接中的图片，请等待处理完成。`
    : `正在处理 ${pendingCount} 张图片，请等待推理完成。`);
  results.className = "results-grid empty-state";
  results.innerHTML = "<p>处理中…</p>";

  try {
    const response = await fetch("/api/upscale", {
      method: "POST",
      body: formData,
    });
    const rawText = await response.text();
    let payload = {};

    try {
      payload = rawText ? JSON.parse(rawText) : {};
    } catch {
      payload = { error: rawText || "服务端返回了无法解析的响应。" };
    }

    if (!response.ok) {
      if (payload.failures?.length) {
        setStatus("error", payload.error || "处理失败，请检查下方详情。");
        renderResults(payload);
        return;
      }
      throw new Error(payload.error || "处理失败。");
    }

    const failureCount = (payload.failures ?? []).length;
    const successCount = (payload.results ?? []).length;
    const warningCount = (payload.warnings ?? []).length;
    const failureText = failureCount ? `，失败 ${failureCount} 张` : "";
    const warningText = warningCount ? `，自动回退 ${warningCount} 次` : "";
    const statusKind = successCount > 0 ? "success" : "error";
    const runtimeText =
      payload.runtime_profile === "cuda"
        ? "CUDA"
        : payload.runtime_profile === "metal"
        ? "Metal(MPS)"
        : payload.runtime_profile === "metal-fallback"
          ? "Metal(MPS) -> CPU"
          : payload.runtime_profile === "cpu-multicore"
            ? `CPU(${payload.cpu_threads || "?"} 线程)`
          : payload.device;

    setStatus(
      statusKind,
      `已完成 ${successCount} 张图片处理${failureText}${warningText}。结果保存在 ${payload.output_dir}，当前推理设备：${runtimeText}。`,
    );
    renderResults(payload);

    if (sourceMode === "input-folder") {
      await loadInputFolderFiles();
    }
  } catch (error) {
    setStatus("error", error.message || "处理失败，请检查服务日志。");
    results.className = "results-grid empty-state";
    results.innerHTML = "<p>处理失败，未生成输出。</p>";
  } finally {
    processing = false;
    submitButton.disabled = false;
  }
});

syncProcessingUI();
loadRuntimeInfo();
syncModeUI();
