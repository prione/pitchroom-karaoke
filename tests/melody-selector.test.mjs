import assert from "node:assert/strict";
import test from "node:test";

import { fusePredominantMelody, selectPredominantMelody } from "../app/melody-selector.mjs";

test("follows a coherent melody instead of the loud bass notes", () => {
  const events = [];
  for (const [index, pitch] of [69, 71, 69, 72, 71].entries()) {
    events.push({ startTimeSeconds: index * 0.4, durationSeconds: 0.36, pitchMidi: pitch, amplitude: 0.72 });
    events.push({ startTimeSeconds: index * 0.4, durationSeconds: 0.38, pitchMidi: 45 + index % 2 * 5, amplitude: 0.92 });
  }

  assert.deepEqual(selectPredominantMelody(events).map(note => note.midi), [69, 71, 69, 72, 71]);
});

test("preserves three short notes inside the opening figure", () => {
  const events = [
    { startTimeSeconds: 0.05, durationSeconds: 0.38, pitchMidi: 69, amplitude: 0.7 },
    { startTimeSeconds: 0.45, durationSeconds: 0.38, pitchMidi: 71, amplitude: 0.74 },
    { startTimeSeconds: 0.85, durationSeconds: 0.36, pitchMidi: 69, amplitude: 0.71 },
  ];

  const result = selectPredominantMelody(events);
  assert.deepEqual(result.map(note => note.midi), [69, 71, 69]);
  assert.deepEqual(result.map(note => note.time), [0.05, 0.45, 0.85]);
});

test("does not chase a brief octave harmonic", () => {
  const events = [
    { startTimeSeconds: 0, durationSeconds: 0.18, pitchMidi: 80, amplitude: 0.7 },
    { startTimeSeconds: 0.18, durationSeconds: 0.12, pitchMidi: 92, amplitude: 0.75 },
    { startTimeSeconds: 0.3, durationSeconds: 0.2, pitchMidi: 80, amplitude: 0.72 },
  ];

  assert.ok(selectPredominantMelody(events).every(note => note.midi === 80));
});

test("uses the perceptual contour instead of loud accompaniment voices", () => {
  const events = [];
  for (const [index, pitch] of [69, 71, 72, 71].entries()) {
    const time = index * 0.4;
    events.push({ startTimeSeconds: time, durationSeconds: 0.35, pitchMidi: pitch, amplitude: 0.62 });
    events.push({ startTimeSeconds: time, durationSeconds: 0.38, pitchMidi: 45, amplitude: 0.95 });
    events.push({ startTimeSeconds: time, durationSeconds: 0.25, pitchMidi: 84, amplitude: 0.8 });
  }
  const contour = [69, 71, 72, 71].map((midi, index) => ({
    time: index * 0.4, duration: 0.35, midi, confidence: 0.8, attack: true,
  }));

  assert.deepEqual(fusePredominantMelody(events, contour).map(note => note.midi), [69, 71, 72, 71]);
});

test("fills piano attacks and ornaments around a smoothed perceptual contour", () => {
  const events = [
    { startTimeSeconds: 0.1, durationSeconds: 0.25, pitchMidi: 89, amplitude: 0.7 },
    { startTimeSeconds: 0.4, durationSeconds: 0.2, pitchMidi: 87, amplitude: 0.68 },
    { startTimeSeconds: 0.7, durationSeconds: 0.28, pitchMidi: 89, amplitude: 0.72 },
    { startTimeSeconds: 1.0, durationSeconds: 0.3, pitchMidi: 94, amplitude: 0.71 },
  ];
  const contour = [{ time: 0.7, duration: 0.6, midi: 89, confidence: 0.8, attack: true }];

  assert.deepEqual(fusePredominantMelody(events, contour).map(note => note.midi), [89, 87, 89, 94]);
});

test("rejects a brief contour excursion into another register", () => {
  const events = [
    { startTimeSeconds: 0, durationSeconds: 0.4, pitchMidi: 89, amplitude: 0.7 },
    { startTimeSeconds: 0.4, durationSeconds: 0.15, pitchMidi: 58, amplitude: 0.85 },
    { startTimeSeconds: 0.6, durationSeconds: 0.4, pitchMidi: 87, amplitude: 0.72 },
  ];
  const contour = [
    { time: 0, duration: 0.35, midi: 89, confidence: 0.8, attack: true },
    { time: 0.4, duration: 0.15, midi: 58, confidence: 0.7, attack: true },
    { time: 0.6, duration: 0.35, midi: 87, confidence: 0.8, attack: true },
  ];

  assert.deepEqual(fusePredominantMelody(events, contour).map(note => note.midi), [89, 87]);
});
