import type { NoteEventTime } from "@spotify/basic-pitch";

import { fusePredominantMelody } from "./melody-selector.mjs";

type MelodyNote = { time: number; duration: number; midi: number; confidence: number; attack: boolean };
type ProgressCallback = (progress: number) => void;

const LOCAL_API = "http://127.0.0.1:8765";
const MODEL_URL = `${LOCAL_API}/api/models/basic-pitch/model.json`;
let modelPromise: Promise<import("@spotify/basic-pitch").BasicPitch> | null = null;
const TRANSCRIPTION_CHUNK_SECONDS = 30;

async function getModel() {
  if (!modelPromise) {
    modelPromise = import("@spotify/basic-pitch").then(({ BasicPitch }) => new BasicPitch(MODEL_URL));
  }
  return modelPromise;
}

async function decodeMono22050(audioUrl: string) {
  const response = await fetch(audioUrl);
  if (!response.ok) throw new Error("AI採譜用の音源を読み込めませんでした");
  const decodeContext = new AudioContext();
  const decoded = await decodeContext.decodeAudioData(await response.arrayBuffer());
  const frameCount = Math.max(1, Math.ceil(decoded.duration * 22_050));
  const offline = new OfflineAudioContext(1, frameCount, 22_050);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  const samples = new Float32Array(rendered.getChannelData(0));
  await decodeContext.close();
  return samples;
}

async function evaluateInChunks(
  model: import("@spotify/basic-pitch").BasicPitch,
  samples: Float32Array,
  frames: number[][],
  onsets: number[][],
  onProgress: ProgressCallback,
) {
  const tf = await import("@tensorflow/tfjs");
  await model.model;
  const chunkSamples = 22_050 * TRANSCRIPTION_CHUNK_SECONDS;
  for (let offset = 0; offset < samples.length; offset += chunkSamples) {
    const chunk = samples.subarray(offset, Math.min(samples.length, offset + chunkSamples));
    tf.engine().startScope();
    try {
      await model.evaluateModel(
        chunk,
        (frameBatch, onsetBatch) => {
          frames.push(...frameBatch);
          onsets.push(...onsetBatch);
        },
        progress => onProgress(Math.min(1, (offset + progress * chunk.length) / samples.length)),
      );
    } finally {
      // Basic Pitch does not dispose its intermediate tensors. Keeping them
      // for an entire song eventually exhausts Electron's WebGL context.
      tf.engine().endScope();
    }
  }
  onProgress(1);
}

export async function extractPredominantMelody(
  audioUrl: string,
  contourNotes: MelodyNote[],
  onProgress: ProgressCallback,
): Promise<MelodyNote[]> {
  const [{ outputToNotesPoly, noteFramesToTime }, samples] = await Promise.all([
    import("@spotify/basic-pitch"),
    decodeMono22050(audioUrl),
  ]);
  const frames: number[][] = [];
  const onsets: number[][] = [];
  const tf = await import("@tensorflow/tfjs");
  try {
    await evaluateInChunks(await getModel(), samples, frames, onsets, onProgress);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!/shader|webgl|texture|gpu/i.test(message)) throw error;

    // Some Windows GPU drivers reject TensorFlow's generated shader. Retry on
    // CPU so instrumental analysis remains functional on those machines.
    frames.length = 0;
    onsets.length = 0;
    const failedModel = await getModel();
    (await failedModel.model).dispose();
    modelPromise = null;
    await tf.setBackend("cpu");
    await tf.ready();
    onProgress(0);
    await evaluateInChunks(await getModel(), samples, frames, onsets, onProgress);
  }
  const events: NoteEventTime[] = noteFramesToTime(outputToNotesPoly(
    frames,
    onsets,
    0.22,
    0.18,
    3,
    true,
    2_100,
    80,
    false,
    11,
  ));
  return fusePredominantMelody(events, contourNotes);
}

export function synthesizeMelodyGuide(notes: MelodyNote[], durationSeconds: number) {
  const sampleRate = 22_050;
  const sampleCount = Math.max(1, Math.ceil(durationSeconds * sampleRate));
  const mix = new Float32Array(sampleCount);
  for (const note of notes) {
    const start = Math.max(0, Math.floor(note.time * sampleRate));
    const length = Math.min(sampleCount - start, Math.ceil(note.duration * sampleRate));
    const frequency = 440 * 2 ** ((note.midi - 69) / 12);
    for (let index = 0; index < length; index++) {
      const elapsed = index / sampleRate;
      const remaining = (length - index) / sampleRate;
      const envelope = Math.min(1, elapsed / 0.018, remaining / 0.045);
      const phase = 2 * Math.PI * frequency * elapsed;
      mix[start + index] += envelope * (Math.sin(phase) * 0.19 + Math.sin(phase * 2) * 0.035);
    }
  }

  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);
  const writeText = (offset: number, text: string) => [...text].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
  writeText(0, "RIFF"); view.setUint32(4, 36 + sampleCount * 2, true); writeText(8, "WAVE");
  writeText(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeText(36, "data"); view.setUint32(40, sampleCount * 2, true);
  for (let index = 0; index < sampleCount; index++) {
    const sample = Math.max(-1, Math.min(1, mix[index]));
    view.setInt16(44 + index * 2, Math.round(sample * 32_767), true);
  }
  return URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
}
