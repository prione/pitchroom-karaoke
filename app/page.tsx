"use client";

import { ChangeEvent, DragEvent, forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { extractPredominantMelody, synthesizeMelodyGuide } from "./basic-pitch-melody";
import { detachMediaSession, isCurrentPlayback } from "./media-session.mjs";
import { getMicAudioConstraints, getMicGateSettings } from "./mic-settings.mjs";
import { simplifyPitchBars } from "./pitch-bars.mjs";

type NotePoint = { time: number; duration?: number; midi: number; confidence: number; attack?: boolean };
type SourceKind = "none" | "file" | "youtube";
type AnalysisMode = "vocal" | "instrumental" | null;
type Theme = "light" | "dark";
type MicDeviceOption = { deviceId: string; label: string };
type AnalysisResult = { mode: "vocal" | "instrumental"; duration: number; notes: NotePoint[]; predominant_notes?: NotePoint[]; instrumental_method?: "adaptive_modules" | "hybrid_fallback"; vocal_ratio: number; youtube_id?: string; youtube_title?: string };
type AnalysisJob = { status: "queued" | "processing" | "completed" | "failed"; progress: number; message: string; result?: AnalysisResult };
type LibraryEntrySummary = { id: string; title: string; source_kind: "file" | "youtube"; youtube_url: string; youtube_id: string; mode: "vocal" | "instrumental"; duration: number; created_at: string; updated_at: string; last_played_at: string | null; trashed_at: string | null; size_bytes: number };
type LibraryEntry = LibraryEntrySummary & { notes: NotePoint[] };
type LibraryStats = { active: number; trashed: number; cache_bytes: number; upload_bytes: number };
type LibraryCleanupPreview = { items: number; reclaimable_bytes: number };
type LibraryOptimizationPreview = { items: number; current_bytes: number };
type YouTubePlayerHandle = { play: (time: number, rate: number) => void; pause: () => void; seekTo: (time: number) => void; setRate: (rate: number) => void };

const LOCAL_API = "http://127.0.0.1:8765";

const NOTE_NAMES = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"];
function midiToHz(midi: number) { return 440 * 2 ** ((midi - 69) / 12); }
function hzToMidi(hz: number) { return 69 + 12 * Math.log2(hz / 440); }
function noteLabel(midi: number) { const rounded = Math.round(midi); return `${NOTE_NAMES[(rounded % 12 + 12) % 12]}${Math.floor(rounded / 12) - 1}`; }
function formatTime(seconds: number) { const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0; return `${Math.floor(safe / 60)}:${Math.floor(safe % 60).toString().padStart(2, "0")}`; }
function formatBytes(bytes: number) { if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`; if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MB`; return `${Math.round(bytes / 1024)} KB`; }
function formatLibraryDate(value: string | null) { if (!value) return "未使用"; return new Intl.DateTimeFormat("ja-JP", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value)); }

function autoCorrelate(buffer: Float32Array, sampleRate: number) {
  let mean = 0;
  for (const sample of buffer) mean += sample;
  mean /= buffer.length;
  let rms = 0;
  for (const sample of buffer) rms += (sample - mean) * (sample - mean);
  rms = Math.sqrt(rms / buffer.length);
  if (rms < 0.0015) return { pitch: -1, clarity: 0, rms };
  const minLag = Math.floor(sampleRate / 1000);
  const maxLag = Math.min(Math.floor(sampleRate / 70), Math.floor(buffer.length / 2));
  let bestLag = -1;
  let best = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let correlation = 0;
    let energyA = 0;
    let energyB = 0;
    for (let i = 0; i < buffer.length - lag; i++) {
      const a = buffer[i] - mean;
      const b = buffer[i + lag] - mean;
      correlation += a * b;
      energyA += a * a;
      energyB += b * b;
    }
    correlation /= Math.sqrt(energyA * energyB) + 1e-9;
    if (correlation > best) { best = correlation; bestLag = lag; }
  }
  return bestLag > 0 ? { pitch: sampleRate / bestLag, clarity: Math.min(1, best), rms } : { pitch: -1, clarity: 0, rms };
}

const YouTubeFrame = forwardRef<YouTubePlayerHandle, { url: string }>(function YouTubeFrame({ url }, ref) {
  const frameRef = useRef<HTMLIFrameElement>(null);
  const id = useMemo(() => {
    try {
      const parsed = new URL(url);
      if (parsed.hostname.includes("youtu.be")) return parsed.pathname.slice(1).split("?")[0];
      if (parsed.pathname.includes("/shorts/")) return parsed.pathname.split("/shorts/")[1]?.split("/")[0];
      return parsed.searchParams.get("v") ?? "";
    } catch { return ""; }
  }, [url]);

  const command = useCallback((func: string, args: unknown[] = []) => {
    frameRef.current?.contentWindow?.postMessage(JSON.stringify({ event: "command", func, args }), "https://www.youtube.com");
  }, []);

  useImperativeHandle(ref, () => ({
    play: (time, rate) => {
      command("mute");
      command("seekTo", [Math.max(0, time), true]);
      command("setPlaybackRate", [rate]);
      command("playVideo");
    },
    pause: () => command("pauseVideo"),
    seekTo: time => command("seekTo", [Math.max(0, time), true]),
    setRate: rate => command("setPlaybackRate", [rate]),
  }), [command]);

  if (!id) return null;
  const origin = typeof window === "undefined" ? "http://localhost:3000" : window.location.origin;
  return <iframe ref={frameRef} className="youtube-frame" src={`https://www.youtube.com/embed/${id}?enablejsapi=1&origin=${encodeURIComponent(origin)}&rel=0&controls=0&disablekb=1&playsinline=1&mute=1`} title="YouTube player（映像のみ）" allow="autoplay; encrypted-media; picture-in-picture" allowFullScreen />;
});

export default function Home() {
  const audioRef = useRef<HTMLAudioElement>(null);
  const guideAudioRef = useRef<HTMLAudioElement>(null);
  const youtubePlayerRef = useRef<YouTubePlayerHandle>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micContextRef = useRef<AudioContext | null>(null);
  const micFrameRef = useRef<number | null>(null);
  const micRequestRef = useRef(0);
  const micLabelsPrimedRef = useRef(false);
  const micDeviceLabelsRef = useRef(new Map<string, string>());
  const micSensitivityRef = useRef(50);
  const canvasFrameRef = useRef<number | null>(null);
  const currentTimeRef = useRef(0);
  const liveMidiRef = useRef<number | null>(null);
  const analysisRequestRef = useRef(0);
  const playbackCommandRef = useRef(0);
  const micOctaveCorrectionRef = useRef(true);
  const viewCenterRef = useRef(66);
  const viewTargetCenterRef = useRef(66);
  const scoringRef = useRef<{ playing: boolean; target: NotePoint | null; keyShift: number }>({ playing: false, target: null, keyShift: 0 });
  const [sourceKind, setSourceKind] = useState<SourceKind>("none");
  const [sourceName, setSourceName] = useState("曲を選択してください");
  const [libraryEntries, setLibraryEntries] = useState<LibraryEntrySummary[]>([]);
  const [currentLibraryId, setCurrentLibraryId] = useState("");
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [libraryQuery, setLibraryQuery] = useState("");
  const [librarySource, setLibrarySource] = useState<"all" | "file" | "youtube">("all");
  const [librarySort, setLibrarySort] = useState<"recent" | "title" | "updated" | "size">("recent");
  const [libraryTrash, setLibraryTrash] = useState(false);
  const [libraryTotal, setLibraryTotal] = useState(0);
  const [libraryHasMore, setLibraryHasMore] = useState(false);
  const [libraryStats, setLibraryStats] = useState<LibraryStats>({ active: 0, trashed: 0, cache_bytes: 0, upload_bytes: 0 });
  const [libraryBusy, setLibraryBusy] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState("");
  const [cleanupPreview, setCleanupPreview] = useState<LibraryCleanupPreview | null>(null);
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const [optimizationPreview, setOptimizationPreview] = useState<LibraryOptimizationPreview | null>(null);
  const [optimizationBusy, setOptimizationBusy] = useState(false);
  const [sourceUrl, setSourceUrl] = useState("");
  const [guideUrl, setGuideUrl] = useState("");
  const [youtubeInput, setYoutubeInput] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(100);
  const [analysisMessage, setAnalysisMessage] = useState("ローカル解析エンジン準備完了");
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>(null);
  const [notes, setNotes] = useState<NotePoint[]>([]);
  const [micActive, setMicActive] = useState(false);
  const [micDevices, setMicDevices] = useState<MicDeviceOption[]>([]);
  const [selectedMicId, setSelectedMicId] = useState("");
  const [micSensitivity, setMicSensitivity] = useState(50);
  const [micSettingsHydrated, setMicSettingsHydrated] = useState(false);
  const [liveMidi, setLiveMidi] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [streak, setStreak] = useState(0);
  const [samples, setSamples] = useState(0);
  const [vocalGuide, setVocalGuide] = useState(true);
  const [guideVolume, setGuideVolume] = useState(72);
  const [keyShift, setKeyShift] = useState(0);
  const [tempo, setTempo] = useState(100);
  const [micOctaveCorrection, setMicOctaveCorrection] = useState(true);
  const [theme, setTheme] = useState<Theme>("light");
  const [toast, setToast] = useState("");

  const targetNote = useMemo(() => {
    if (!notes.length) return null;
    return notes.reduce((best, note) => Math.abs(note.time - currentTime) < Math.abs(best.time - currentTime) ? note : best, notes[0]);
  }, [notes, currentTime]);
  const accuracy = samples ? Math.round(score / samples) : 0;

  const fetchLibraryPage = useCallback(async (offset = 0, append = false) => {
    setLibraryBusy(true);
    try {
      const params = new URLSearchParams({
        q: libraryQuery, source: librarySource, sort: librarySort,
        trashed: String(libraryTrash), offset: String(offset), limit: "50",
      });
      const response = await fetch(`${LOCAL_API}/api/library?${params}`);
      if (!response.ok) return;
      const payload = await response.json() as { entries?: LibraryEntrySummary[]; total?: number; has_more?: boolean };
      setLibraryEntries(current => append ? [...current, ...(payload.entries ?? [])] : (payload.entries ?? []));
      setLibraryTotal(payload.total ?? 0);
      setLibraryHasMore(payload.has_more ?? false);
    } catch {
      // The analysis engine may still be starting; a completed analysis retries this.
    } finally {
      setLibraryBusy(false);
    }
  }, [libraryQuery, librarySource, librarySort, libraryTrash]);

  const refreshLibraryStats = useCallback(async () => {
    try {
      const response = await fetch(`${LOCAL_API}/api/library-stats`);
      if (response.ok) setLibraryStats(await response.json() as LibraryStats);
    } catch { /* The analysis engine may still be starting. */ }
  }, []);

  const refreshOptimizationPreview = useCallback(async () => {
    try {
      const response = await fetch(`${LOCAL_API}/api/library-optimize`);
      if (response.ok) setOptimizationPreview(await response.json() as LibraryOptimizationPreview);
    } catch { /* The analysis engine may still be starting. */ }
  }, []);

  const refreshLibrary = useCallback(async () => {
    await Promise.all([fetchLibraryPage(0, false), refreshLibraryStats()]);
  }, [fetchLibraryPage, refreshLibraryStats]);

  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(""), 2600); return () => window.clearTimeout(timer); }, [toast]);
  useEffect(() => { const timer = window.setTimeout(() => void fetchLibraryPage(0, false), libraryOpen ? 180 : 0); return () => window.clearTimeout(timer); }, [fetchLibraryPage, libraryOpen]);
  useEffect(() => { if (!libraryOpen) return; const timer = window.setTimeout(() => void Promise.all([refreshLibraryStats(), refreshOptimizationPreview()]), 0); return () => window.clearTimeout(timer); }, [libraryOpen, refreshLibraryStats, refreshOptimizationPreview]);
  useEffect(() => {
    if (!libraryOpen) return;
    const close = (event: KeyboardEvent) => { if (event.key === "Escape") setLibraryOpen(false); };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [libraryOpen]);
  useEffect(() => { currentTimeRef.current = currentTime; }, [currentTime]);
  useEffect(() => { liveMidiRef.current = liveMidi; }, [liveMidi]);
  useEffect(() => { scoringRef.current = { playing, target: targetNote, keyShift }; }, [playing, targetNote, keyShift]);
  useEffect(() => { micOctaveCorrectionRef.current = micOctaveCorrection; }, [micOctaveCorrection]);
  useEffect(() => {
    const storedSensitivityValue = window.localStorage.getItem("pitchroom-mic-sensitivity-v1");
    const storedSensitivity = storedSensitivityValue === null ? Number.NaN : Number(storedSensitivityValue);
    const storedDevice = window.localStorage.getItem("pitchroom-mic-device");
    window.queueMicrotask(() => {
      if (Number.isFinite(storedSensitivity) && storedSensitivity >= 0 && storedSensitivity <= 100) setMicSensitivity(storedSensitivity);
      if (storedDevice) setSelectedMicId(storedDevice);
      setMicSettingsHydrated(true);
    });

    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices?.enumerateDevices) return;
    let disposed = false;
    const refreshDevices = async () => {
      let inputs = (await mediaDevices.enumerateDevices()).filter(device => device.kind === "audioinput");
      if (!micLabelsPrimedRef.current && inputs.some(device => !device.label)) {
        micLabelsPrimedRef.current = true;
        let labelStream: MediaStream | null = null;
        try {
          labelStream = await mediaDevices.getUserMedia({ audio: getMicAudioConstraints() });
          inputs = (await mediaDevices.enumerateDevices()).filter(device => device.kind === "audioinput");
          const labelTrack = labelStream.getAudioTracks()[0];
          const labelDeviceId = labelTrack?.getSettings().deviceId;
          if (labelTrack?.label && labelDeviceId) micDeviceLabelsRef.current.set(labelDeviceId, labelTrack.label);
          if (labelTrack?.label && inputs.length === 1) micDeviceLabelsRef.current.set(inputs[0].deviceId, labelTrack.label);
          if (labelTrack?.label && inputs.some(device => device.deviceId === "default")) micDeviceLabelsRef.current.set("default", labelTrack.label);
        } catch {
          // The selector remains usable with generic labels if Windows denies access.
        } finally {
          labelStream?.getTracks().forEach(track => track.stop());
        }
      }
      if (disposed) return;
      const options = inputs.map(device => ({ deviceId: device.deviceId, label: device.label || micDeviceLabelsRef.current.get(device.deviceId) || "" }));
      setMicDevices(options);
      setSelectedMicId(current => {
        if (current && inputs.some(device => device.deviceId === current)) return current;
        return (inputs.find(device => device.deviceId === "default") ?? inputs[0])?.deviceId ?? "";
      });
    };
    void refreshDevices();
    mediaDevices.addEventListener("devicechange", refreshDevices);
    return () => { disposed = true; mediaDevices.removeEventListener("devicechange", refreshDevices); };
  }, []);
  useEffect(() => { micSensitivityRef.current = micSensitivity; if (micSettingsHydrated) window.localStorage.setItem("pitchroom-mic-sensitivity-v1", String(micSensitivity)); }, [micSensitivity, micSettingsHydrated]);
  useEffect(() => { if (micSettingsHydrated) window.localStorage.setItem("pitchroom-mic-device", selectedMicId); }, [selectedMicId, micSettingsHydrated]);
  useEffect(() => {
    const anchor = notes.find(note => note.time >= currentTimeRef.current) ?? notes[0];
    if (!anchor) return;
    const center = Math.round((anchor.midi + keyShift) / 12) * 12;
    viewCenterRef.current = center;
    viewTargetCenterRef.current = center;
  }, [notes, keyShift]);
  useEffect(() => () => { if (sourceUrl.startsWith("blob:")) URL.revokeObjectURL(sourceUrl); }, [sourceUrl]);
  useEffect(() => () => { if (guideUrl.startsWith("blob:")) URL.revokeObjectURL(guideUrl); }, [guideUrl]);
  useEffect(() => () => { micRequestRef.current += 1; micStreamRef.current?.getTracks().forEach(track => track.stop()); void micContextRef.current?.close(); if (micFrameRef.current) cancelAnimationFrame(micFrameRef.current); if (canvasFrameRef.current) cancelAnimationFrame(canvasFrameRef.current); }, []);

  const stopPlayback = useCallback(() => {
    playbackCommandRef.current += 1;
    const audio = audioRef.current;
    const guide = guideAudioRef.current;
    detachMediaSession(audio, guide, youtubePlayerRef.current);
    setPlaying(false);
    setCurrentTime(0);
    currentTimeRef.current = 0;
  }, []);

  const waitForAnalysis = useCallback(async (jobId: string, kind: "file" | "youtube", requestId: number) => {
    while (true) {
      await new Promise(resolve => window.setTimeout(resolve, 650));
      if (analysisRequestRef.current !== requestId) return false;
      const statusResponse = await fetch(`${LOCAL_API}/api/jobs/${jobId}`);
      if (!statusResponse.ok) throw new Error("解析状況を取得できません");
      const job = await statusResponse.json() as AnalysisJob;
      if (analysisRequestRef.current !== requestId) return false;
      setAnalysisProgress(job.progress ?? 0);
      setAnalysisMessage(job.message ?? "解析中です");
      if (job.status === "failed") throw new Error(job.message);
      if (job.status !== "completed" || !job.result) continue;

      const cacheBuster = Date.now();
      const nextSourceUrl = `${LOCAL_API}/api/jobs/${jobId}/audio/accompaniment?v=${cacheBuster}`;
      let nextGuideUrl: string;
      let melodyNotes = job.result.notes;
      if (job.result.mode === "instrumental" && !melodyNotes.length) {
        setAnalysisProgress(72);
        setAnalysisMessage("AIで全音符を採譜し、知覚上の主旋律を選んでいます");
        melodyNotes = await extractPredominantMelody(
          `${LOCAL_API}/api/jobs/${jobId}/audio/melody-source?v=${cacheBuster}`,
          (job.result.predominant_notes ?? []).map(note => ({ ...note, duration: note.duration ?? 0.1, attack: note.attack ?? true })),
          progress => {
            if (analysisRequestRef.current !== requestId) return;
            setAnalysisProgress(72 + Math.round(progress * 27));
            setAnalysisMessage(`AI採譜から主旋律を抽出しています ${Math.round(progress * 100)}%`);
          },
        );
        if (analysisRequestRef.current !== requestId) return false;
        if (!melodyNotes.length) throw new Error("AI採譜から主旋律を選択できませんでした");
        nextGuideUrl = synthesizeMelodyGuide(
          melodyNotes.map(note => ({ ...note, duration: note.duration ?? 0.1, attack: note.attack ?? false })),
          job.result.duration,
        );
      } else if (job.result.mode === "vocal") {
        nextGuideUrl = `${LOCAL_API}/api/jobs/${jobId}/audio/vocals?v=${cacheBuster}`;
      } else {
        if (!melodyNotes.length) throw new Error("高解像度採譜から主旋律を選択できませんでした");
        nextGuideUrl = synthesizeMelodyGuide(
          melodyNotes.map(note => ({ ...note, duration: note.duration ?? 0.1, attack: note.attack ?? true })),
          job.result.duration,
        );
      }
      if (analysisRequestRef.current !== requestId) {
        if (nextGuideUrl.startsWith("blob:")) URL.revokeObjectURL(nextGuideUrl);
        return false;
      }
      // Commit the source, guide, and notes together. Exposing the new audio
      // before instrumental transcription completes can leave the stage blank
      // while a playable source is already mounted.
      setSourceUrl(nextSourceUrl);
      setGuideUrl(nextGuideUrl);
      const finalNotes = simplifyPitchBars(melodyNotes, job.result.mode === "instrumental");
      setNotes(finalNotes);
      setDuration(job.result.duration);
      setAnalysisMode(job.result.mode);
      setAnalysisProgress(100);
      if (kind === "youtube" && job.result.youtube_title) setSourceName(job.result.youtube_title);
      try {
        const savedResponse = await fetch(`${LOCAL_API}/api/library`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job_id: jobId, notes: finalNotes }),
        });
        if (savedResponse.ok) {
          const saved = await savedResponse.json() as LibraryEntry;
          setCurrentLibraryId(saved.id);
          await refreshLibrary();
        }
      } catch {
        // Playback remains available even if library indexing is temporarily unavailable.
      }
      setToast(job.result.mode === "vocal" ? "ボーカルを分離し、歌声から主旋律を抽出しました" : "全音符をAI採譜し、知覚上の主旋律を抽出しました");
      return true;
    }
  }, [refreshLibrary]);

  const showAnalysisError = useCallback((error: unknown, youtube = false) => {
    setNotes([]);
    setAnalysisProgress(0);
    const message = error instanceof Error ? error.message : "解析に失敗しました";
    setAnalysisMessage(message);
    const engineUnavailable = /failed to fetch|networkerror|network request/i.test(message);
    setToast(youtube
      ? `YouTube動画を解析できませんでした: ${message}`
      : engineUnavailable ? "ローカル解析エンジンに接続できません" : `解析に失敗しました: ${message}`);
  }, []);

  const analyzeLocalAudio = useCallback(async (file: File, requestId: number) => {
    setLoading(true);
    setAnalysisProgress(1);
    setAnalysisMessage("ローカル解析エンジンへ音源を渡しています");
    setAnalysisMode(null);
    setNotes([]);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const createdResponse = await fetch(`${LOCAL_API}/api/jobs`, { method: "POST", body: form });
      if (!createdResponse.ok) throw new Error("ローカル解析エンジンに接続できません");
      const created = await createdResponse.json() as { id: string };
      await waitForAnalysis(created.id, "file", requestId);
    } catch (error) {
      if (analysisRequestRef.current === requestId) showAnalysisError(error);
    } finally {
      if (analysisRequestRef.current === requestId) setLoading(false);
    }
  }, [showAnalysisError, waitForAnalysis]);

  const loadFile = useCallback((file?: File) => {
    if (!file || !file.type.startsWith("audio/")) { setToast("MP3・WAV・M4Aなどの音源を選択してください"); return; }
    const requestId = ++analysisRequestRef.current;
    stopPlayback();
    setCurrentLibraryId(""); setSourceKind("file"); setSourceName(file.name); setSourceUrl(""); setGuideUrl(""); setYoutubeUrl("");
    setCurrentTime(0); setDuration(0); setPlaying(false); setNotes([]); setScore(0); setSamples(0); setStreak(0);
    setAnalysisMessage("音源を準備しています"); setAnalysisMode(null);
    void analyzeLocalAudio(file, requestId);
  }, [analyzeLocalAudio, stopPlayback]);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); loadFile(event.dataTransfer.files[0]); };
  const handleFile = (event: ChangeEvent<HTMLInputElement>) => loadFile(event.target.files?.[0]);
  const loadYoutube = async () => {
    if (!/(youtube\.com|youtu\.be)/i.test(youtubeInput)) { setToast("有効なYouTube URLを入力してください"); return; }
    const requestId = ++analysisRequestRef.current;
    stopPlayback();
    setCurrentLibraryId(""); setSourceKind("youtube"); setSourceName("YouTube セッション"); setYoutubeUrl(youtubeInput.trim()); setSourceUrl(""); setGuideUrl("");
    setCurrentTime(0); setPlaying(false); setNotes([]); setDuration(0); setScore(0); setSamples(0); setStreak(0); setAnalysisMode(null);
    setLoading(true); setAnalysisProgress(1); setAnalysisMessage("YouTube動画を確認しています");
    try {
      const createdResponse = await fetch(`${LOCAL_API}/api/youtube-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: youtubeInput.trim() }),
      });
      if (!createdResponse.ok) {
        const body = await createdResponse.json().catch(() => null) as { detail?: string } | null;
        throw new Error(body?.detail ?? "YouTube解析を開始できません");
      }
      const created = await createdResponse.json() as { id: string };
      await waitForAnalysis(created.id, "youtube", requestId);
    } catch (error) {
      if (analysisRequestRef.current === requestId) showAnalysisError(error, true);
    } finally {
      if (analysisRequestRef.current === requestId) setLoading(false);
    }
  };

  const loadLibraryEntry = async (libraryId: string) => {
    if (!libraryId) return;
    const requestId = ++analysisRequestRef.current;
    stopPlayback();
    setLoading(true); setAnalysisProgress(1); setAnalysisMessage("保存済みの解析結果を読み込んでいます");
    try {
      const response = await fetch(`${LOCAL_API}/api/library/${encodeURIComponent(libraryId)}`);
      if (!response.ok) {
        const body = await response.json().catch(() => null) as { detail?: string } | null;
        throw new Error(body?.detail ?? "解析済み曲を読み込めません");
      }
      const entry = await response.json() as LibraryEntry;
      if (analysisRequestRef.current !== requestId) return;
      const cacheBuster = encodeURIComponent(entry.updated_at || entry.id);
      const finalNotes = (entry.notes ?? []).map(note => ({
        ...note,
        duration: note.duration ?? 0.1,
        confidence: note.confidence ?? 1,
      }));
      if (!finalNotes.length) throw new Error("保存済みの音程データがありません。もう一度だけ元音源から解析してください");
      const nextGuideUrl = entry.mode === "vocal"
        ? `${LOCAL_API}/api/library/${encodeURIComponent(entry.id)}/audio/guide?v=${cacheBuster}`
        : synthesizeMelodyGuide(finalNotes, entry.duration);
      setCurrentLibraryId(entry.id);
      setSourceKind(entry.source_kind);
      setSourceName(entry.title);
      setSourceUrl(`${LOCAL_API}/api/library/${encodeURIComponent(entry.id)}/audio/accompaniment?v=${cacheBuster}`);
      setGuideUrl(nextGuideUrl);
      setYoutubeUrl(entry.youtube_url || (entry.youtube_id ? `https://www.youtube.com/watch?v=${entry.youtube_id}` : ""));
      setYoutubeInput(entry.youtube_url || "");
      setNotes(finalNotes);
      setDuration(entry.duration);
      setCurrentTime(0); setPlaying(false); setScore(0); setSamples(0); setStreak(0);
      setAnalysisMode(entry.mode); setAnalysisProgress(100); setAnalysisMessage("保存済みの解析結果を読み込みました");
      setToast(`「${entry.title}」をライブラリから読み込みました`);
      setLibraryOpen(false);
      void refreshLibraryStats();
    } catch (error) {
      if (analysisRequestRef.current === requestId) showAnalysisError(error);
    } finally {
      if (analysisRequestRef.current === requestId) setLoading(false);
    }
  };

  const clearLoadedSong = () => {
    analysisRequestRef.current += 1;
    stopPlayback();
    setCurrentLibraryId(""); setSourceKind("none"); setSourceName("曲を選択してください");
    setSourceUrl(""); setGuideUrl(""); setYoutubeUrl(""); setCurrentTime(0); setDuration(0);
    setNotes([]); setAnalysisMode(null); setAnalysisMessage("ローカル解析エンジン準備完了");
    setScore(0); setSamples(0); setStreak(0);
  };

  const trashLibraryEntry = async (entry: LibraryEntrySummary) => {
    try {
      const response = await fetch(`${LOCAL_API}/api/library/${encodeURIComponent(entry.id)}/trash`, { method: "POST" });
      if (!response.ok) throw new Error("ゴミ箱へ移動できませんでした");
      if (currentLibraryId === entry.id) clearLoadedSong();
      setToast(`「${entry.title}」をゴミ箱へ移動しました`);
      await refreshLibrary();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "ゴミ箱へ移動できませんでした");
    }
  };

  const restoreLibraryEntry = async (entry: LibraryEntrySummary) => {
    try {
      const response = await fetch(`${LOCAL_API}/api/library/${encodeURIComponent(entry.id)}/restore`, { method: "POST" });
      if (!response.ok) throw new Error("曲を復元できませんでした");
      setToast(`「${entry.title}」を復元しました`);
      await refreshLibrary();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "曲を復元できませんでした");
    }
  };

  const permanentlyDeleteLibraryEntry = async (entry: LibraryEntrySummary) => {
    try {
      const response = await fetch(`${LOCAL_API}/api/library/${encodeURIComponent(entry.id)}`, { method: "DELETE" });
      if (!response.ok) throw new Error("曲を完全に削除できませんでした");
      const result = await response.json() as { freed_bytes?: number };
      setDeleteConfirmId("");
      setToast(`「${entry.title}」を削除しました${result.freed_bytes ? `（${formatBytes(result.freed_bytes)} 解放）` : ""}`);
      await refreshLibrary();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "曲を完全に削除できませんでした");
    }
  };

  const previewLibraryCleanup = async () => {
    setCleanupBusy(true);
    try {
      const response = await fetch(`${LOCAL_API}/api/library-cleanup`);
      if (!response.ok) throw new Error("不要データを確認できませんでした");
      const preview = await response.json() as LibraryCleanupPreview;
      if (!preview.items) {
        setCleanupPreview(null);
        setToast("整理できる不要データはありません");
      } else {
        setCleanupPreview(preview);
      }
    } catch (error) {
      setToast(error instanceof Error ? error.message : "不要データを確認できませんでした");
    } finally {
      setCleanupBusy(false);
    }
  };

  const cleanupLibraryCache = async () => {
    setCleanupBusy(true);
    try {
      const response = await fetch(`${LOCAL_API}/api/library-cleanup`, { method: "POST" });
      const payload = await response.json().catch(() => ({})) as { freed_bytes?: number; detail?: string };
      if (!response.ok) throw new Error(payload.detail || "不要データを整理できませんでした");
      setCleanupPreview(null);
      setToast(`${formatBytes(payload.freed_bytes ?? 0)} の不要データを整理しました`);
      await refreshLibraryStats();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "不要データを整理できませんでした");
    } finally {
      setCleanupBusy(false);
    }
  };

  const optimizeLibraryStorage = async () => {
    setOptimizationBusy(true);
    try {
      const response = await fetch(`${LOCAL_API}/api/library-optimize`, { method: "POST" });
      const payload = await response.json().catch(() => ({})) as { items?: number; failed_items?: number; freed_bytes?: number; detail?: string };
      if (!response.ok) throw new Error(payload.detail || "保存曲を圧縮できませんでした");
      const failure = payload.failed_items ? `、${payload.failed_items}曲は未処理` : "";
      setToast(`${payload.items ?? 0}曲を圧縮し、${formatBytes(payload.freed_bytes ?? 0)}解放しました${failure}`);
      await Promise.all([refreshLibrary(), refreshOptimizationPreview()]);
    } catch (error) {
      setToast(error instanceof Error ? error.message : "保存曲を圧縮できませんでした");
    } finally {
      setOptimizationBusy(false);
    }
  };

  const releaseMicResources = () => {
    micStreamRef.current?.getTracks().forEach(track => track.stop());
    micStreamRef.current = null;
    if (micFrameRef.current !== null) cancelAnimationFrame(micFrameRef.current);
    micFrameRef.current = null;
    const context = micContextRef.current;
    micContextRef.current = null;
    if (context && context.state !== "closed") void context.close();
    liveMidiRef.current = null;
    setLiveMidi(null);
  };

  const connectMic = async (deviceId: string, switching = false) => {
    const requestId = ++micRequestRef.current;
    releaseMicResources();
    setMicActive(false);
    let stream: MediaStream | null = null;
    try {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: getMicAudioConstraints(deviceId) });
      } catch (error) {
        const unavailable = error instanceof DOMException && (error.name === "NotFoundError" || error.name === "OverconstrainedError");
        if (!deviceId || !unavailable) throw error;
        stream = await navigator.mediaDevices.getUserMedia({ audio: getMicAudioConstraints() });
      }
      if (requestId !== micRequestRef.current) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      const context = new AudioContext();
      await context.resume();
      const source = context.createMediaStreamSource(stream);
      const highPass = context.createBiquadFilter(); highPass.type = "highpass"; highPass.frequency.value = 75; highPass.Q.value = 0.7;
      const lowPass = context.createBiquadFilter(); lowPass.type = "lowpass"; lowPass.frequency.value = 1400; lowPass.Q.value = 0.7;
      const analyser = context.createAnalyser(); analyser.fftSize = 2048; analyser.smoothingTimeConstant = 0.1;
      source.connect(highPass).connect(lowPass).connect(analyser);
      micStreamRef.current = stream; micContextRef.current = context; setMicActive(true);
      const activeTrack = stream.getAudioTracks()[0];
      const activeDeviceId = activeTrack?.getSettings().deviceId;
      if (activeTrack?.label && activeDeviceId) micDeviceLabelsRef.current.set(activeDeviceId, activeTrack.label);
      if (activeTrack?.label && deviceId) micDeviceLabelsRef.current.set(deviceId, activeTrack.label);
      const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === "audioinput");
      if (requestId !== micRequestRef.current) return;
      if (activeTrack?.label && inputs.length === 1) micDeviceLabelsRef.current.set(inputs[0].deviceId, activeTrack.label);
      setMicDevices(inputs.map(device => ({ deviceId: device.deviceId, label: device.label || micDeviceLabelsRef.current.get(device.deviceId) || "" })));
      if (activeDeviceId && inputs.some(device => device.deviceId === activeDeviceId)) setSelectedMicId(activeDeviceId);
      setToast(switching ? `使用マイクを「${activeTrack?.label || "既定のマイク"}」に切り替えました` : `「${activeTrack?.label || "既定のマイク"}」を接続しました`);
      const buffer = new Float32Array(analyser.fftSize);
      const candidates: number[] = [];
      let noiseFloor = 0.006;
      let lastVoiceAt = 0;
      let lastUiUpdate = 0;
      let lastScoreAt = 0;
      const calibrationUntil = performance.now() + 600;
      const tick = (now: number) => {
        analyser.getFloatTimeDomainData(buffer);
        const result = autoCorrelate(buffer, context.sampleRate);
        if (now < calibrationUntil) {
          noiseFloor = Math.min(0.025, noiseFloor * 0.9 + result.rms * 0.1);
          micFrameRef.current = requestAnimationFrame(tick);
          return;
        }
        const gateSettings = getMicGateSettings(micSensitivityRef.current);
        const gate = Math.max(gateSettings.minimumRms, noiseFloor * gateSettings.noiseMultiplier);
        const isVoice = result.pitch >= 75 && result.pitch <= 1200 && result.clarity >= gateSettings.minimumClarity && result.rms >= gate;
        if (!isVoice) {
          if (result.rms < gate) noiseFloor = noiseFloor * 0.985 + result.rms * 0.015;
          candidates.length = 0;
          if (now - lastVoiceAt > 130 && liveMidiRef.current !== null) {
            liveMidiRef.current = null;
            setLiveMidi(null);
          }
          micFrameRef.current = requestAnimationFrame(tick);
          return;
        }

        const detectedMidi = hzToMidi(result.pitch);
        candidates.push(detectedMidi);
        if (candidates.length > 5) candidates.shift();
        const ordered = [...candidates].sort((a, b) => a - b);
        const stableMidi = ordered[Math.floor(ordered.length / 2)];
        const spread = ordered.at(-1)! - ordered[0];
        if (candidates.length < 3 || spread > 0.9) {
          micFrameRef.current = requestAnimationFrame(tick);
          return;
        }

        const scoring = scoringRef.current;
        const targetMidi = scoring.target ? scoring.target.midi + scoring.keyShift : null;
        const octaveOffset = micOctaveCorrectionRef.current && targetMidi !== null
          ? Math.max(-2, Math.min(2, Math.round((targetMidi - stableMidi) / 12))) * 12
          : 0;
        const correctedMidi = stableMidi + octaveOffset;
        lastVoiceAt = now;
        liveMidiRef.current = correctedMidi;
        if (now - lastUiUpdate >= 80) { lastUiUpdate = now; setLiveMidi(correctedMidi); }
        if (scoring.playing && scoring.target && now - lastScoreAt >= 50) {
          lastScoreAt = now;
          const cents = Math.abs((correctedMidi - (scoring.target.midi + scoring.keyShift)) * 100);
          const points = Math.max(0, 100 - cents * 0.8);
          setScore(value => value + points); setSamples(value => value + 1); setStreak(value => cents < 45 ? value + 1 : 0);
        }
        micFrameRef.current = requestAnimationFrame(tick);
      };
      micFrameRef.current = requestAnimationFrame(tick);
    } catch {
      stream?.getTracks().forEach(track => track.stop());
      if (requestId !== micRequestRef.current) return;
      releaseMicResources();
      setMicActive(false);
      setToast("選択したマイクを使用できません。接続と許可を確認してください");
    }
  };

  const startMic = () => {
    if (micActive) {
      micRequestRef.current += 1;
      releaseMicResources();
      setMicActive(false);
      setToast("マイクを停止しました");
      return;
    }
    void connectMic(selectedMicId);
  };

  const selectMic = (deviceId: string) => {
    setSelectedMicId(deviceId);
    if (micActive) void connectMic(deviceId, true);
  };

  const seekPlayback = (time: number) => {
    const next = Math.max(0, Math.min(duration, time));
    setCurrentTime(next);
    currentTimeRef.current = next;
    if (audioRef.current) audioRef.current.currentTime = next;
    if (guideAudioRef.current) guideAudioRef.current.currentTime = next;
    if (sourceKind === "youtube") youtubePlayerRef.current?.seekTo(next);
  };

  const togglePlayback = async () => {
    if (loading) { setToast("主旋律と伴奏を解析しています"); return; }
    if (sourceKind === "none") { setToast("先に音源を読み込んでください"); return; }
    const audio = audioRef.current;
    if (audio) {
      const command = ++playbackCommandRef.current;
      audio.playbackRate = tempo / 100;
      const guide = guideAudioRef.current;
      if (guide) {
        guide.playbackRate = tempo / 100;
        guide.volume = vocalGuide ? guideVolume / 100 : 0;
        guide.currentTime = audio.currentTime;
      }
      if (playing) {
        audio.pause();
        guide?.pause();
        if (sourceKind === "youtube") youtubePlayerRef.current?.pause();
      } else {
        try {
          await audio.play();
        } catch {
          // Changing songs intentionally aborts a pending play() promise.
          if (!isCurrentPlayback(command, playbackCommandRef.current, audio, audioRef.current)) return;
          setToast("音源を再生できませんでした");
          return;
        }
        if (!isCurrentPlayback(command, playbackCommandRef.current, audio, audioRef.current)) {
          audio.pause();
          return;
        }
        if (guide && vocalGuide && guideVolume > 0) {
          await guide.play().catch(() => undefined);
          if (!isCurrentPlayback(command, playbackCommandRef.current, audio, audioRef.current)) {
            audio.pause();
            guide.pause();
            return;
          }
        }
        if (sourceKind === "youtube") youtubePlayerRef.current?.play(audio.currentTime, tempo / 100);
      }
      if (playbackCommandRef.current === command) setPlaying(!playing);
      return;
    }
    if (sourceKind === "youtube") { setToast("YouTube動画の解析完了を待ってください"); return; }
    setToast("音源の準備が完了していません");
  };

  useEffect(() => {
    const guide = guideAudioRef.current;
    if (!guide) return;
    guide.volume = vocalGuide ? guideVolume / 100 : 0;
    if (!vocalGuide || guideVolume === 0) guide.pause();
    else if (playing && sourceKind !== "none") void guide.play().catch(() => undefined);
  }, [vocalGuide, guideVolume, playing, sourceKind]);

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    let width = 0;
    let height = 0;
    const syncCanvasSize = () => {
      const nextWidth = Math.max(1, canvas.clientWidth);
      const nextHeight = Math.max(1, canvas.clientHeight);
      if (nextWidth === width && nextHeight === height) return;
      width = nextWidth;
      height = nextHeight;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    syncCanvasSize();
    const visibleSeconds = 12;
    let lastUiUpdate = 0;
    let lastFrameTime = 0;

    const draw = (frameTime: number) => {
      syncCanvasSize();
      const centerX = width * 0.28;
      let visualTime = currentTimeRef.current;
      if (playing && sourceKind !== "none" && audioRef.current) {
        visualTime = audioRef.current.currentTime;
        currentTimeRef.current = visualTime;
        if (frameTime - lastUiUpdate >= 50) {
          lastUiUpdate = frameTime;
          setCurrentTime(visualTime);
        }
      }

      const activeNote = notes.find(note => visualTime >= note.time - 0.03 && visualTime <= note.time + (note.duration ?? 0.1) + 0.03);
      const nextNote = notes.find(note => note.time > visualTime);
      const upperTrigger = viewTargetCenterRef.current + 10;
      const lowerTrigger = viewTargetCenterRef.current - 10;
      const lookAheadSeconds = 1.05;
      const boundaryNote = notes.find(note => {
        if (note.time < visualTime || note.time > visualTime + lookAheadSeconds) return false;
        const pitch = note.midi + keyShift;
        return pitch > upperTrigger || pitch < lowerTrigger;
      });
      const focusNote = boundaryNote ?? activeNote ?? nextNote;
      if (focusNote) {
        const focusPitch = focusNote.midi + keyShift;
        const activePitch = activeNote ? activeNote.midi + keyShift : null;
        const canMoveUp = activePitch === null || activePitch >= viewTargetCenterRef.current - 1;
        const canMoveDown = activePitch === null || activePitch <= viewTargetCenterRef.current + 1;
        if (focusPitch > upperTrigger && canMoveUp) viewTargetCenterRef.current += 12;
        else if (focusPitch < lowerTrigger && canMoveDown) viewTargetCenterRef.current -= 12;
      }
      const elapsed = lastFrameTime ? Math.min(50, frameTime - lastFrameTime) : 16;
      const smoothing = 1 - Math.exp(-elapsed / 480);
      viewCenterRef.current += (viewTargetCenterRef.current - viewCenterRef.current) * smoothing;
      const minMidi = viewCenterRef.current - 12;
      const maxMidi = viewCenterRef.current + 12;
      lastFrameTime = frameTime;

      ctx.fillStyle = "#141513"; ctx.fillRect(0, 0, width, height);
      for (let midi = Math.ceil(minMidi); midi <= maxMidi; midi++) {
        const y = height - ((midi - minMidi) / (maxMidi - minMidi)) * height;
        ctx.strokeStyle = midi % 12 === 0 ? "#393b35" : "#242621"; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }
      ctx.fillStyle = "#8d9087";
      ctx.font = "9px ui-monospace, monospace";
      ctx.textAlign = "right";
      ctx.fillText(noteLabel(Math.round(maxMidi)), width - 8, 13);
      ctx.fillText(noteLabel(Math.round(minMidi)), width - 8, height - 7);
      notes.forEach(note => {
        const x = centerX + ((note.time - visualTime) / visibleSeconds) * width;
        if (x < -80 || x > width + 80) return;
        const shiftedMidi = note.midi + keyShift;
        if (shiftedMidi < minMidi - 1 || shiftedMidi > maxMidi + 1) return;
        const y = height - ((shiftedMidi - minMidi) / (maxMidi - minMidi)) * height;
        const barWidth = Math.max(8, (note.duration ?? 0.18) / visibleSeconds * width);
        ctx.fillStyle = note.time < visualTime ? "#5f6b54" : "#a8ff63"; ctx.beginPath(); ctx.roundRect(x, y - 5, barWidth, 10, 5); ctx.fill();
      });
      ctx.strokeStyle = "#f2c44d"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(centerX, 0); ctx.lineTo(centerX, height); ctx.stroke();
      const visualMidi = liveMidiRef.current;
      if (visualMidi !== null) {
        const y = height - ((visualMidi - minMidi) / (maxMidi - minMidi)) * height;
        ctx.fillStyle = "#ff6e52"; ctx.shadowColor = "#ff6e52"; ctx.shadowBlur = 12; ctx.beginPath(); ctx.arc(centerX, y, 7, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
      }
      canvasFrameRef.current = requestAnimationFrame(draw);
    };

    canvasFrameRef.current = requestAnimationFrame(draw);
    return () => { if (canvasFrameRef.current) cancelAnimationFrame(canvasFrameRef.current); };
  }, [notes, keyShift, playing, sourceKind]);

  return (
    <main data-theme={theme}>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="PitchRoom home"><img className="brand-mark" src="/favicon.svg" alt="" /><span>PITCHROOM</span></a>
        <div className="session-status"><span className={micActive ? "status-dot live" : "status-dot"} />{micActive ? "MIC LIVE" : "STUDIO READY"}</div>
        <button className="icon-button theme-button" onClick={() => setTheme(value => value === "light" ? "dark" : "light")} aria-label={theme === "light" ? "ダークモードへ切り替え" : "ライトモードへ切り替え"} title={theme === "light" ? "ダークモード" : "ライトモード"}>{theme === "light" ? "◐" : "☀"}</button>
      </header>

      <section className="workspace" id="top">
        <aside className="source-panel">
          <div className="current-source">
            <span className="source-art">♫</span>
            <div>
              <small>{sourceKind === "none" ? "CURRENT SONG" : "NOW LOADED"}</small>
              <button className="current-song-button" onClick={() => setLibraryOpen(true)} aria-label="解析済み曲ライブラリを開く"><span>{sourceName}</span><i>⌄</i></button>
              <em>{sourceKind === "file" ? "LOCAL AUDIO" : sourceKind === "youtube" ? "YOUTUBE" : "音源を選択してください"}</em>
            </div>
          </div>

          <div className="dropzone" onDragOver={event => event.preventDefault()} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()} role="button" tabIndex={0} onKeyDown={event => { if (event.key === "Enter") fileInputRef.current?.click(); }}>
            <input ref={fileInputRef} type="file" accept="audio/*,.m4a" onChange={handleFile} hidden />
            <span className="upload-icon">↥</span><strong>音源ファイルを選択 / ドロップ</strong><small>MP3 / WAV / M4A · 最大500MB</small>
          </div>
          <div className="or"><span />OR<span /></div>
          <label className="url-label" htmlFor="youtube">YOUTUBE URL</label>
          <div className="url-row"><input id="youtube" value={youtubeInput} onChange={event => setYoutubeInput(event.target.value)} placeholder="https://youtube.com/watch?v=..." /><button onClick={loadYoutube} disabled={loading}>{loading && sourceKind === "youtube" ? "解析中" : "読み込む"}</button></div>
          <p className="rights-note">保存・解析が許可された動画のみご利用ください。</p>
          <div className="analysis"><span>LOCAL MELODY ENGINE</span><b>{loading ? `${analysisProgress}%` : analysisProgress === 100 ? "READY" : "OFFLINE"}</b><div><i style={{ width: `${analysisProgress}%` }} /></div><p>{analysisMessage}{analysisMode ? ` — ${analysisMode === "vocal" ? "ボーカル分離＋歌声ゲート" : "インスト主旋律モード"}` : ""}</p></div>

          <section className="side-controls" aria-labelledby="session-controls-title">
            <h3 id="session-controls-title">カラオケ設定</h3>
            <div className="control-block"><label>キー変更 <output>{keyShift > 0 ? `+${keyShift}` : keyShift}</output></label><div className="stepper"><button onClick={() => setKeyShift(value => Math.max(-6, value - 1))}>−</button><span>{keyShift === 0 ? "原曲キー" : `${keyShift > 0 ? "+" : ""}${keyShift} 半音`}</span><button onClick={() => setKeyShift(value => Math.min(6, value + 1))}>＋</button></div></div>
            <div className="control-block"><label>テンポ <output>{tempo}%</output></label><input type="range" min="70" max="120" value={tempo} onChange={event => { const rate = Number(event.target.value) / 100; setTempo(Number(event.target.value)); if (audioRef.current) audioRef.current.playbackRate = rate; if (guideAudioRef.current) guideAudioRef.current.playbackRate = rate; if (sourceKind === "youtube") youtubePlayerRef.current?.setRate(rate); }} /></div>
            <div className="control-block"><label>ガイド音量 <output>{guideVolume}%</output></label><input type="range" min="0" max="100" value={guideVolume} onChange={event => setGuideVolume(Number(event.target.value))} /></div>
            <div className="control-block mic-device-block">
              <label htmlFor="mic-device">使用マイク <output>{micActive ? "LIVE" : ""}</output></label>
              <select id="mic-device" value={selectedMicId} onChange={event => selectMic(event.target.value)} disabled={!micDevices.length}>
                {!micDevices.length && <option value="">既定のマイク</option>}
                {micDevices.map((device, index) => <option key={device.deviceId} value={device.deviceId}>{device.label || (device.deviceId === "default" ? "システム既定" : `マイク ${index + 1}`)}</option>)}
              </select>
            </div>
            <div className="control-block"><label htmlFor="mic-sensitivity">マイク感度 <output>{micSensitivity}%</output></label><input id="mic-sensitivity" type="range" min="0" max="100" step="1" value={micSensitivity} onChange={event => setMicSensitivity(Number(event.target.value))} title="高いほど小さい声にも反応します" /></div>
            <button className="toggle-row" onClick={() => setVocalGuide(value => !value)}><span><strong>主旋律ガイド</strong><small>解析したメロディを再生</small></span><i className={vocalGuide ? "toggle on" : "toggle"} /></button>
            <button className="toggle-row" onClick={() => setMicOctaveCorrection(value => !value)}><span><strong>マイクのオクターブ補正</strong><small>基準音に近い音域へ自動調整</small></span><i className={micOctaveCorrection ? "toggle on" : "toggle"} /></button>
          </section>
        </aside>

        <section className="stage">
          <div className="stage-head"><div><span className="eyebrow">02 — SING</span><h2>{sourceName}</h2><p>{sourceKind === "youtube" ? "動画はミュートで同期再生し、解析した伴奏と主旋律ガイドを鳴らします" : sourceKind === "file" ? "主旋律ガイドに声を重ねてください" : "左側から音源ファイルまたはYouTube動画を読み込んでください"}</p></div><div className="score"><small>LIVE SCORE</small><strong>{accuracy || "—"}</strong><span>/ 100</span></div></div>
          {sourceKind === "youtube" && youtubeUrl && <YouTubeFrame ref={youtubePlayerRef} url={youtubeUrl} />}
          <div className={sourceKind === "youtube" && youtubeUrl ? "pitch-shell youtube-pitch" : "pitch-shell"}>
              <div className="pitch-label"><span>HIGH</span><span>{targetNote ? noteLabel(targetNote.midi + keyShift) : "—"}</span><span>LOW</span></div>
              <canvas ref={canvasRef} className="pitch-canvas" aria-label="主旋律とマイク音程を表示するピッチバー" />
              <div className="legend"><span><i className="guide-dot" />主旋律</span><span><i className="voice-dot" />あなたの声</span><b>{micActive ? liveMidi ? `${noteLabel(liveMidi)} / ${midiToHz(liveMidi).toFixed(0)}Hz` : "声を待っています" : "マイク未接続"}</b></div>
          </div>

          {sourceKind !== "none" && sourceUrl && <>
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <audio ref={audioRef} src={sourceUrl} onTimeUpdate={event => { const time = event.currentTarget.currentTime; setCurrentTime(time); const guide = guideAudioRef.current; if (guide && Math.abs(guide.currentTime - time) > 0.12) guide.currentTime = time; }} onLoadedMetadata={event => setDuration(event.currentTarget.duration)} onEnded={() => { setPlaying(false); guideAudioRef.current?.pause(); youtubePlayerRef.current?.pause(); }} />
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <audio ref={guideAudioRef} src={guideUrl} preload="auto" />
          </>}
          <div className="transport">
            <button className="skip" onClick={() => seekPlayback(currentTime - 5)} aria-label="5秒戻る">↶<small>5</small></button>
            <button className="play" onClick={togglePlayback} aria-label={playing ? "停止" : "再生"}>{playing ? "Ⅱ" : "▶"}</button>
            <button className="skip" onClick={() => seekPlayback(currentTime + 5)} aria-label="5秒進む">↷<small>5</small></button>
            <span className="time">{formatTime(currentTime)}</span><input className="seek" type="range" min="0" max={duration || 1} value={Math.min(currentTime, duration || 1)} onChange={event => seekPlayback(Number(event.target.value))} /><span className="time">{formatTime(duration)}</span>
          </div>

          <div className="performance-strip">
            <button className={micActive ? "mic-button active" : "mic-button"} onClick={startMic}><span>●</span>{micActive ? "マイクを停止" : "マイクを接続"}</button>
            <div><small>COMBO</small><strong>{streak}<em>x</em></strong></div><div><small>TARGET</small><strong>{targetNote ? noteLabel(targetNote.midi + keyShift) : "—"}</strong></div><div><small>YOUR PITCH</small><strong>{liveMidi ? noteLabel(liveMidi) : "—"}</strong></div>
          </div>
        </section>

      </section>
      <footer><span>PITCHROOM / PERSONAL KARAOKE STUDIO</span><span>音源は端末内で処理され、アップロードされません。</span></footer>
      {libraryOpen && <div className="library-overlay" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) { setLibraryOpen(false); setCleanupPreview(null); } }}>
        <section className="library-dialog" role="dialog" aria-modal="true" aria-labelledby="library-title">
          <header className="library-head">
            <div><span className="eyebrow">LOCAL SONG LIBRARY</span><h2 id="library-title">解析済みの曲</h2><p>{libraryStats.active}曲 · 使用領域 {formatBytes(libraryStats.cache_bytes + libraryStats.upload_bytes)}</p></div>
            <button className="library-close" onClick={() => { setLibraryOpen(false); setCleanupPreview(null); }} aria-label="ライブラリを閉じる">×</button>
          </header>
          <div className="library-toolbar">
            <label className="library-search"><span>⌕</span><input value={libraryQuery} onChange={event => setLibraryQuery(event.target.value)} placeholder="曲名を検索" /></label>
            <div className="library-source-filter" aria-label="音源種別">
              {(["all", "file", "youtube"] as const).map(value => <button key={value} className={librarySource === value ? "active" : ""} onClick={() => setLibrarySource(value)}>{value === "all" ? "すべて" : value === "file" ? "音源" : "YouTube"}</button>)}
            </div>
            <select className="library-sort" aria-label="並べ替え" value={librarySort} onChange={event => setLibrarySort(event.target.value as typeof librarySort)}>
              <option value="recent">最近使用</option><option value="title">曲名順</option><option value="updated">解析日時</option><option value="size">容量順</option>
            </select>
            <button className={libraryTrash ? "trash-tab active" : "trash-tab"} onClick={() => { setLibraryTrash(value => !value); setDeleteConfirmId(""); }}>ゴミ箱 {libraryStats.trashed ? `(${libraryStats.trashed})` : ""}</button>
          </div>
          <div className="library-storage">
            <span>再生音源はFLACで可逆圧縮できます。ゴミ箱の曲は不要データ整理の対象外です。</span>
            <div className="storage-actions">
              <button disabled={optimizationBusy || !optimizationPreview?.items} onClick={() => void optimizeLibraryStorage()}>{optimizationBusy ? "圧縮中…" : optimizationPreview?.items ? `保存曲を圧縮 (${optimizationPreview.items})` : "圧縮済み"}</button>
              {cleanupPreview
                ? <div className="cleanup-confirm"><strong>{formatBytes(cleanupPreview.reclaimable_bytes)} · {cleanupPreview.items}件を整理可能</strong><button disabled={cleanupBusy} onClick={() => void cleanupLibraryCache()}>{cleanupBusy ? "整理中…" : "整理する"}</button><button disabled={cleanupBusy} onClick={() => setCleanupPreview(null)}>取消</button></div>
                : <button disabled={cleanupBusy} onClick={() => void previewLibraryCleanup()}>{cleanupBusy ? "確認中…" : "不要データを確認"}</button>}
            </div>
          </div>
          <div className="library-result-head"><span>{libraryTrash ? "ゴミ箱" : "ライブラリ"} · {libraryTotal}件</span>{libraryBusy && <i>更新中…</i>}</div>
          <div className="library-list">
            {!libraryBusy && !libraryEntries.length && <div className="library-empty"><strong>{libraryTrash ? "ゴミ箱は空です" : "該当する曲がありません"}</strong><small>{libraryTrash ? "削除した曲はここから復元できます" : "検索条件を変えるか、新しい曲を解析してください"}</small></div>}
            {libraryEntries.map(entry => <article className={currentLibraryId === entry.id ? "library-item current" : "library-item"} key={entry.id}>
              <button className="library-item-main" disabled={libraryTrash} onClick={() => void loadLibraryEntry(entry.id)}>
                <span className={entry.source_kind === "youtube" ? "library-art youtube" : "library-art"}>{entry.source_kind === "youtube" ? "▶" : "♫"}</span>
                <span className="library-copy"><strong>{entry.title}</strong><small>{entry.source_kind === "youtube" ? "YouTube" : "ローカル音源"} · {entry.mode === "vocal" ? "ボーカル" : "インスト"} · {formatTime(entry.duration)}</small></span>
                <span className="library-meta"><strong>{formatBytes(entry.size_bytes)}</strong><small>{libraryTrash ? `削除 ${formatLibraryDate(entry.trashed_at)}` : `使用 ${formatLibraryDate(entry.last_played_at)}`}</small></span>
              </button>
              <div className="library-actions">
                {libraryTrash ? <>
                  <button className="restore-button" onClick={() => void restoreLibraryEntry(entry)}>復元</button>
                  {deleteConfirmId === entry.id
                    ? <span className="delete-confirm"><small>元に戻せません</small><button onClick={() => void permanentlyDeleteLibraryEntry(entry)}>削除する</button><button onClick={() => setDeleteConfirmId("")}>取消</button></span>
                    : <button className="delete-button" onClick={() => setDeleteConfirmId(entry.id)}>完全削除</button>}
                </> : <button className="trash-button" onClick={() => void trashLibraryEntry(entry)} aria-label={`${entry.title}をゴミ箱へ移動`} title="ゴミ箱へ移動">⌫</button>}
              </div>
            </article>)}
          </div>
          {libraryHasMore && <button className="library-more" disabled={libraryBusy} onClick={() => void fetchLibraryPage(libraryEntries.length, true)}>さらに50件を表示</button>}
        </section>
      </div>}
      {toast && <div className="toast" role="status">{toast}</div>}
    </main>
  );
}
