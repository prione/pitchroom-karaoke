const fs = require("node:fs");
const { Essentia, EssentiaWASM } = require("essentia.js");
const wav = require(require.resolve("node-wav", { paths: [require.resolve("essentia.js")] }));

const [inputPath, outputPath, minimumFrequencyArgument] = process.argv.slice(2);
const minimumFrequency = Number(minimumFrequencyArgument ?? 80);
if (!inputPath || !outputPath) {
  throw new Error("Usage: node extract-melodia.cjs <input.wav> <output.json>");
}

const decoded = wav.decode(fs.readFileSync(inputPath));
const mono = new Float32Array(decoded.channelData[0].length);
for (const channel of decoded.channelData) {
  for (let index = 0; index < mono.length; index++) mono[index] += channel[index] / decoded.channelData.length;
}

const essentia = new Essentia(EssentiaWASM);
const chunkSeconds = 30;
const overlapSeconds = 4;
const chunkSamples = Math.round(chunkSeconds * decoded.sampleRate);
const stepSamples = Math.round((chunkSeconds - overlapSeconds) * decoded.sampleRate);
const notes = [];
try {
  for (let offset = 0; offset < mono.length; offset += stepSamples) {
    const end = Math.min(mono.length, offset + chunkSamples);
    const chunk = mono.subarray(offset, end);
    const signal = essentia.arrayToVector(chunk);
    let equalized;
    let melody;
    let segmented;
    try {
      equalized = essentia.EqualLoudness(signal, decoded.sampleRate).signal;
      melody = essentia.PredominantPitchMelodia(
        equalized, 10, 3, 2048, false, 0.8, 128, 1, 40, 2_100, 80, minimumFrequency,
        20, 0.9, 0.9, 27.5625, 55, decoded.sampleRate, 120, false, 0.2,
      );
      segmented = essentia.PitchContourSegmentation(
        melody.pitch, equalized, 128, 0.14, 120, -1.5, decoded.sampleRate, 440,
      );
      const onset = essentia.vectorToArray(segmented.onset);
      const duration = essentia.vectorToArray(segmented.duration);
      const midi = essentia.vectorToArray(segmented.MIDIpitch);
      const frameConfidence = [...essentia.vectorToArray(melody.pitchConfidence)];
      const hopSeconds = 128 / decoded.sampleRate;
      const offsetSeconds = offset / decoded.sampleRate;
      const keepStart = offset === 0 ? 0 : overlapSeconds / 2;
      const keepEnd = end === mono.length ? chunk.length / decoded.sampleRate : chunkSeconds - overlapSeconds / 2;
      for (let index = 0; index < onset.length; index++) {
        if (onset[index] < keepStart || onset[index] >= keepEnd) continue;
        const startFrame = Math.max(0, Math.floor(onset[index] / hopSeconds));
        const endFrame = Math.min(frameConfidence.length, Math.ceil((onset[index] + duration[index]) / hopSeconds));
        const confidence = frameConfidence.slice(startFrame, endFrame).filter(value => value > 0).sort();
        notes.push({
          time: Math.round((offsetSeconds + onset[index]) * 10_000) / 10_000,
          duration: Math.round(duration[index] * 10_000) / 10_000,
          midi: Math.round(midi[index]),
          confidence: Math.round((confidence[Math.floor(confidence.length / 2)] ?? 0.5) * 10_000) / 10_000,
          attack: true,
        });
      }
    } finally {
      segmented?.onset?.delete();
      segmented?.duration?.delete();
      segmented?.MIDIpitch?.delete();
      melody?.pitch?.delete();
      melody?.pitchConfidence?.delete();
      equalized?.delete();
      signal.delete();
    }
    if (end === mono.length) break;
  }
  fs.writeFileSync(outputPath, JSON.stringify(notes));
} finally {
  essentia.shutdown();
}
