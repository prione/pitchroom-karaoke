import { readFile, writeFile } from "node:fs/promises";

import { BasicPitch, noteFramesToTime, outputToNotesPoly } from "@spotify/basic-pitch";

import { selectPredominantMelody } from "../app/melody-selector.mjs";

const rawPath = process.argv[2];
const eventOutputPath = process.argv[3];
const modelUrl = "http://127.0.0.1:8765/api/models/basic-pitch/model.json";
if (!rawPath) throw new Error("Usage: node scripts/validate-basic-pitch.mjs <mono-22050-f32le>");

const bytes = await readFile(rawPath);
const samples = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
const model = new BasicPitch(modelUrl);
const frames = [];
const onsets = [];
const contours = [];
await model.evaluateModel(
  samples,
  (frameBatch, onsetBatch, contourBatch) => {
    frames.push(...frameBatch);
    onsets.push(...onsetBatch);
    contours.push(...contourBatch);
  },
  progress => {
    const percentage = Math.round(progress * 100);
    if (percentage % 10 === 0) process.stderr.write(`\rBasic Pitch ${percentage}%`);
  },
);
process.stderr.write("\n");

const events = noteFramesToTime(outputToNotesPoly(frames, onsets, 0.22, 0.18, 3, true, 2_100, 80, false, 11));
if (eventOutputPath) await writeFile(eventOutputPath, JSON.stringify(events));
const melody = selectPredominantMelody(events);
const jumps = melody.slice(1).map((note, index) => Math.abs(note.midi - melody[index].midi)).sort((a, b) => a - b);
const pitches = melody.map(note => note.midi).sort((a, b) => a - b);
const quantile = (values, fraction) => values[Math.min(values.length - 1, Math.floor(values.length * fraction))] ?? null;

console.log(JSON.stringify({
  eventCount: events.length,
  melodyCount: melody.length,
  openingEvents: events.filter(note => note.startTimeSeconds < 1.3),
  openingMelody: melody.filter(note => note.time < 1.3),
  pitchRange: [quantile(pitches, 0.05), quantile(pitches, 0.95)],
  jumpMedian: quantile(jumps, 0.5),
  jumpP95: quantile(jumps, 0.95),
}, null, 2));
