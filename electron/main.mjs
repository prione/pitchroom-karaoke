import { app, BrowserWindow, dialog, Menu, session, shell } from "electron";
import { spawn, spawnSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFile, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const electronDir = path.dirname(fileURLToPath(import.meta.url));
const sourceProjectRoot = path.resolve(electronDir, "..");
const applicationRoot = app.isPackaged ? app.getAppPath() : sourceProjectRoot;
const resourcesRoot = app.isPackaged ? process.resourcesPath : sourceProjectRoot;
const backendRoot = app.isPackaged ? path.join(resourcesRoot, "backend") : sourceProjectRoot;
const runtimeRoot = app.isPackaged ? path.join(resourcesRoot, "runtime") : sourceProjectRoot;
const rendererRoot = path.join(applicationRoot, "dist-renderer");
const appIconPath = path.join(applicationRoot, "assets", "pitchroom-icon.ico");
const uiUrl = "http://127.0.0.1:3000";
const apiHealthUrl = "http://127.0.0.1:8765/api/health";
const developmentMode = process.argv.includes("--dev");
const ownedProcesses = new Set();
let mainWindow = null;
let logStream = null;
let staticServer = null;

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) app.quit();

function createLogStream() {
  const logDir = path.join(app.getPath("userData"), "logs");
  mkdirSync(logDir, { recursive: true });
  logStream = createWriteStream(path.join(logDir, "pitchroom.log"), { flags: "a" });
  logStream.write(`\n--- ${new Date().toISOString()} PitchRoom start ---\n`);
}

function log(message) {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  logStream?.write(line);
  if (developmentMode) process.stdout.write(line);
}

function attachProcessLogging(child, label) {
  child.stdout?.on("data", chunk => log(`${label}: ${String(chunk).trimEnd()}`));
  child.stderr?.on("data", chunk => log(`${label}: ${String(chunk).trimEnd()}`));
  child.on("error", error => log(`${label} process error: ${error.stack ?? error.message}`));
  child.on("exit", (code, signal) => {
    log(`${label} stopped (code=${code}, signal=${signal})`);
    ownedProcesses.delete(child);
  });
}

function startProcess(command, args, label, extraEnv = {}, workingDirectory = sourceProjectRoot) {
  const child = spawn(command, args, {
    cwd: workingDirectory,
    env: { ...process.env, ...extraEnv },
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  ownedProcesses.add(child);
  attachProcessLogging(child, label);
  return child;
}

const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"], [".js", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"], [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml"], [".png", "image/png"], [".ico", "image/x-icon"],
  [".woff2", "font/woff2"], [".wasm", "application/wasm"],
]);

function startStaticInterface() {
  if (staticServer) return Promise.resolve();
  if (!existsSync(path.join(rendererRoot, "index.html"))) {
    throw new Error("配布用の画面がありません。pnpm run build:rendererを実行してください。");
  }
  staticServer = createServer((request, response) => {
    const requestedUrl = new URL(request.url ?? "/", uiUrl);
    const relativePath = decodeURIComponent(requestedUrl.pathname === "/" ? "index.html" : requestedUrl.pathname.slice(1));
    let filePath = path.resolve(rendererRoot, relativePath);
    if (path.relative(rendererRoot, filePath).startsWith("..")) {
      response.writeHead(403).end();
      return;
    }
    if (!existsSync(filePath) || !statSync(filePath).isFile()) filePath = path.join(rendererRoot, "index.html");
    readFile(filePath, (error, data) => {
      if (error) {
        response.writeHead(500).end("PITCHROOM interface could not be loaded");
        return;
      }
      response.writeHead(200, {
        "Content-Type": contentTypes.get(path.extname(filePath).toLowerCase()) ?? "application/octet-stream",
        "Cache-Control": filePath.endsWith("index.html") ? "no-store" : "public, max-age=31536000, immutable",
      });
      response.end(data);
    });
  });
  return new Promise((resolve, reject) => {
    staticServer.once("error", reject);
    staticServer.listen(3000, "127.0.0.1", () => {
      log(`Static interface started from ${rendererRoot}`);
      resolve();
    });
  });
}

function isInside(candidate, root) {
  const relative = path.relative(root, candidate);
  return relative !== "" && !relative.startsWith("..") && !path.isAbsolute(relative);
}

function removeUpdateDirectory(candidate, updateRoot) {
  if (isInside(candidate, updateRoot) && existsSync(candidate)) rmSync(candidate, { recursive: true, force: true });
}

function youtubeRuntimeVersion(python, overrides) {
  const result = spawnSync(python, ["-c", "import yt_dlp; print(yt_dlp.version.__version__)"], {
    windowsHide: true,
    encoding: "utf8",
    timeout: 10_000,
    env: { ...process.env, PYTHONPATH: overrides, PYTHONNOUSERSITE: "1" },
  });
  return result.status === 0 ? result.stdout.trim() : "unknown";
}

function updateYoutubeRuntime(python) {
  if (!app.isPackaged || process.env.PITCHROOM_DISABLE_YTDLP_UPDATE === "1") return "";
  const updateRoot = path.join(app.getPath("userData"), "updates");
  const overrides = path.join(updateRoot, "python-overrides");
  const staging = path.join(updateRoot, "python-overrides.staging");
  const backup = path.join(updateRoot, "python-overrides.backup");
  const marker = path.join(updateRoot, "yt-dlp-update.json");
  mkdirSync(updateRoot, { recursive: true });
  try {
    if (existsSync(marker)) {
      const age = Date.now() - statSync(marker).mtimeMs;
      if (age < 24 * 60 * 60 * 1000 && existsSync(overrides)) {
        log(`yt-dlp auto-update is current (${youtubeRuntimeVersion(python, overrides)}).`);
        return overrides;
      }
    }
    removeUpdateDirectory(staging, updateRoot);
    const result = spawnSync(
      python,
      [
        "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-compile",
        "--no-warn-script-location", "--target", staging, "yt-dlp[default]",
      ],
      { windowsHide: true, encoding: "utf8", timeout: 45_000, maxBuffer: 5 * 1024 * 1024 },
    );
    if (result.status !== 0 || !existsSync(path.join(staging, "yt_dlp"))) {
      throw new Error((result.stderr || result.stdout || "update failed").trim().slice(-1600));
    }
    removeUpdateDirectory(backup, updateRoot);
    if (existsSync(overrides)) renameSync(overrides, backup);
    try {
      renameSync(staging, overrides);
    } catch (error) {
      if (existsSync(backup) && !existsSync(overrides)) renameSync(backup, overrides);
      throw error;
    }
    removeUpdateDirectory(backup, updateRoot);
    const version = youtubeRuntimeVersion(python, overrides);
    writeFileSync(marker, JSON.stringify({ checkedAt: new Date().toISOString(), version }), "utf8");
    log(`yt-dlp auto-update completed (${version}).`);
  } catch (error) {
    removeUpdateDirectory(staging, updateRoot);
    log(`yt-dlp auto-update skipped; bundled version remains available: ${error.message}`);
  }
  return existsSync(overrides) ? overrides : "";
}

async function probe(url, validate = () => true) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(1200), cache: "no-store" });
    if (!response.ok) return false;
    return validate(await response.text());
  } catch {
    return false;
  }
}

async function waitForService(url, label, validate, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await probe(url, validate)) return;
    await new Promise(resolve => setTimeout(resolve, 400));
  }
  throw new Error(`${label}の起動を確認できませんでした。`);
}

async function startLocalServices() {
  const python = app.isPackaged
    ? path.join(resourcesRoot, "python-runtime", "python.exe")
    : path.join(sourceProjectRoot, ".venv", "Scripts", "python.exe");
  if (!existsSync(python)) {
    throw new Error(app.isPackaged
      ? "同梱されたPython解析環境が見つかりません。アプリを再インストールしてください。"
      : "Python解析環境がありません。PitchRoom.cmdから初回セットアップを実行してください。");
  }
  const youtubeOverrides = updateYoutubeRuntime(python);
  const dataRoot = app.isPackaged ? path.join(app.getPath("userData"), "data") : path.join(sourceProjectRoot, ".pitchroom");
  const pythonPath = [youtubeOverrides, backendRoot].filter(Boolean).join(path.delimiter);
  const analysisEnvironment = {
    PYTHONUNBUFFERED: "1",
    PYTHONNOUSERSITE: "1",
    PYTHONPATH: pythonPath,
    HF_HUB_DISABLE_SYMLINKS_WARNING: "1",
    PITCHROOM_DATA_ROOT: dataRoot,
    PITCHROOM_APP_ROOT: runtimeRoot,
    PITCHROOM_MODEL_ROOT: app.isPackaged
      ? path.join(runtimeRoot, "basic-pitch-model")
      : path.join(sourceProjectRoot, "node_modules", "@spotify", "basic-pitch", "model"),
    PITCHROOM_NODE_PATH: process.execPath,
    ELECTRON_RUN_AS_NODE: "1",
  };

  const apiAlreadyRunning = await probe(apiHealthUrl, text => {
    try { return JSON.parse(text).local_only === true; } catch { return false; }
  });
  if (!apiAlreadyRunning) {
    startProcess(
      python,
      ["-m", "uvicorn", "local_backend.main:app", "--host", "127.0.0.1", "--port", "8765"],
      "analysis",
      analysisEnvironment,
      backendRoot,
    );
  } else {
    log("Reusing an existing PitchRoom analysis service.");
  }

  const uiAlreadyRunning = await probe(uiUrl, text => text.includes("PITCHROOM"));
  if (!uiAlreadyRunning) {
    if (developmentMode) {
      const cli = path.join(sourceProjectRoot, "node_modules", "vite", "bin", "vite.js");
      if (!existsSync(cli)) throw new Error("画面側の依存関係がありません。pnpm installを実行してください。");
      startProcess(
        process.execPath,
        [cli, "--config", "vite.electron.config.ts", "--host", "127.0.0.1", "--port", "3000"],
        "interface",
        { ELECTRON_RUN_AS_NODE: "1" },
      );
    } else {
      await startStaticInterface();
    }
  } else {
    log("Reusing an existing PitchRoom interface service.");
  }

  await Promise.all([
    waitForService(apiHealthUrl, "音声解析エンジン", text => {
      try { return JSON.parse(text).local_only === true; } catch { return false; }
    }),
    waitForService(uiUrl, "画面", text => text.includes("PITCHROOM")),
  ]);
}

function splashHtml() {
  return `<!doctype html><html lang="ja"><head><meta charset="utf-8"><style>
    :root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#171814;color:#eee9de;font-family:system-ui,-apple-system,sans-serif;display:grid;place-items:center;height:100vh}
    main{text-align:center}.mark{width:64px;height:64px;margin:0 auto 22px;display:block}
    h1{margin:0;font:700 20px ui-monospace,monospace;letter-spacing:.17em}.line{width:240px;height:3px;margin:28px auto 14px;background:#34352f;overflow:hidden}.line:after{content:"";display:block;width:45%;height:100%;background:#a8ff63;animation:move 1.1s ease-in-out infinite alternate}
    p{color:#85877e;font:12px ui-monospace,monospace;letter-spacing:.08em}@keyframes move{to{transform:translateX(125%)}}
  </style></head><body><main><svg class="mark" viewBox="0 0 64 64" aria-hidden="true"><rect x="2" y="2" width="60" height="60" rx="14" fill="#171814"/><path d="M8 19H18M46 19H56M8 32H16M48 32H56M8 45H19M45 45H56" stroke="#a8ff63" stroke-width="4" stroke-linecap="round"/><rect x="24" y="9" width="16" height="30" rx="8" fill="#eee9de"/><path d="M19 29V35C19 42.18 24.82 48 32 48C39.18 48 45 42.18 45 35V29M32 48V55M25 55H39" stroke="#a8ff63" stroke-width="4" stroke-linecap="round"/></svg><h1>PITCHROOM</h1><div class="line"></div><p>LOCAL AUDIO ENGINE STARTING</p></main></body></html>`;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1080,
    minHeight: 700,
    show: false,
    backgroundColor: "#171814",
    title: "PITCHROOM",
    icon: appIconPath,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(electronDir, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      spellcheck: false,
    },
  });
  Menu.setApplicationMenu(null);
  void mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(splashHtml())}`);
  mainWindow.once("ready-to-show", () => mainWindow?.show());
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\//i.test(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("console-message", (event, level, message, line, sourceId) => {
    const detail = typeof event?.message === "string"
      ? event
      : { level, message, lineNumber: line, sourceId };
    log(`renderer[${detail.level ?? "log"}]: ${detail.message ?? ""} (${detail.sourceId ?? "unknown"}:${detail.lineNumber ?? 0})`);
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(uiUrl)) event.preventDefault();
  });
  mainWindow.on("closed", () => { mainWindow = null; });
}

function stopOwnedProcesses() {
  for (const child of [...ownedProcesses]) {
    if (!child.pid || child.killed) continue;
    if (process.platform === "win32") {
      spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
    } else {
      child.kill("SIGTERM");
    }
  }
  ownedProcesses.clear();
  staticServer?.close();
  staticServer = null;
}

app.on("second-instance", () => {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show();
  mainWindow.focus();
});

app.on("before-quit", () => {
  stopOwnedProcesses();
  logStream?.end();
});

app.on("window-all-closed", () => app.quit());

if (gotLock) {
  app.whenReady().then(async () => {
    app.setAppUserModelId("com.pitchroom.karaoke");
    createLogStream();
    const allowedOrigin = new URL(uiUrl).origin;
    session.defaultSession.setPermissionCheckHandler((_webContents, permission, requestingOrigin) => {
      return permission === "media" && requestingOrigin === allowedOrigin;
    });
    session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
      callback(permission === "media" && new URL(webContents.getURL()).origin === allowedOrigin);
    });
    createWindow();
    try {
      await startLocalServices();
      await mainWindow?.loadURL(uiUrl);
      mainWindow?.show();
      log("PitchRoom is ready.");
    } catch (error) {
      log(`Startup failed: ${error.stack ?? error.message}`);
      await dialog.showMessageBox({
        type: "error",
        title: "PITCHROOMを起動できませんでした",
        message: "ローカルサービスの起動に失敗しました。",
        detail: `${error.message}\n\nログ: ${path.join(app.getPath("userData"), "logs", "pitchroom.log")}`,
      });
      app.quit();
    }
  });
}
