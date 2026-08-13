import assert from "node:assert/strict";
import test from "node:test";

import { simplifyPitchBars } from "../app/pitch-bars.mjs";

test("preserves the alternating notes in a fast trill", () => {
  const source = [
    { time: 0.4644, duration: 2.2059, midi: 78, confidence: 0.5948 },
    { time: 2.6703, duration: 0.0929, midi: 79, confidence: 0.567 },
    { time: 2.7632, duration: 0.7663, midi: 78, confidence: 0.5209 },
    { time: 3.5294, duration: 0.1625, midi: 79, confidence: 0.4879 },
    { time: 3.692, duration: 0.1161, midi: 81, confidence: 0.4803 },
    { time: 3.8313, duration: 0.0929, midi: 79, confidence: 0.5062 },
    { time: 3.9242, duration: 3.2044, midi: 78, confidence: 0.6007 },
    { time: 7.1285, duration: 0.1161, midi: 79, confidence: 0.4823 },
    { time: 7.2446, duration: 0.1161, midi: 78, confidence: 0.494 },
    { time: 7.3607, duration: 0.3251, midi: 79, confidence: 0.4851 },
  ];

  assert.deepEqual(simplifyPitchBars(source).map(note => note.midi), [78, 79, 78, 79, 81, 79, 78, 79, 78, 79]);
});

test("collapses a short vocal vibrato turn when ornaments are disabled", () => {
  const source = [
    { time: 0, duration: 0.4, midi: 69, confidence: 0.9 },
    { time: 0.4, duration: 0.12, midi: 70, confidence: 0.8 },
    { time: 0.52, duration: 0.48, midi: 69, confidence: 0.91 },
  ];

  const result = simplifyPitchBars(source, false);

  assert.equal(result.length, 1);
  assert.equal(result[0].midi, 69);
  assert.ok((result[0].duration ?? 0) >= 1);
});

test("still removes a short isolated octave glitch", () => {
  const source = [
    { time: 0, duration: 0.4, midi: 69, confidence: 0.8 },
    { time: 0.4, duration: 0.09, midi: 81, confidence: 0.35 },
    { time: 0.49, duration: 0.4, midi: 69, confidence: 0.82 },
  ];

  const result = simplifyPitchBars(source);
  assert.equal(result.length, 1);
  assert.equal(result[0].midi, 69);
  assert.ok((result[0].duration ?? 0) >= 0.89);
});

test("keeps repeated attacks as separate bars at the same pitch", () => {
  const source = [
    { time: 0, duration: 0.6, midi: 77, confidence: 0.8 },
    { time: 0.6, duration: 0.6, midi: 77, confidence: 0.82, attack: true },
    { time: 1.2, duration: 0.6, midi: 77, confidence: 0.79, attack: true },
  ];

  const result = simplifyPitchBars(source);
  assert.equal(result.length, 3);
  assert.deepEqual(result.map(note => note.time), [0, 0.6, 1.2]);
});
