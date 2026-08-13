/**
 * Select one perceptually predominant voice from polyphonic note events.
 * This is deliberately voice-leading based: the loudest or highest note can
 * win locally, but only a coherent phrase wins over time.
 *
 * @param {{startTimeSeconds:number,durationSeconds:number,pitchMidi:number,amplitude:number}[]} input
 */
export function selectPredominantMelody(input) {
  const notes = input
    .filter(note => Number.isFinite(note.startTimeSeconds) && Number.isFinite(note.pitchMidi))
    .filter(note => note.durationSeconds >= 0.045 && note.pitchMidi >= 40 && note.pitchMidi <= 100)
    .map(note => ({
      time: Math.max(0, note.startTimeSeconds),
      duration: Math.max(0.06, note.durationSeconds),
      midi: Math.round(note.pitchMidi),
      amplitude: Math.max(0, Math.min(1, note.amplitude)),
    }))
    .sort((a, b) => a.time - b.time || b.midi - a.midi);
  if (!notes.length) return [];

  const amplitudes = notes.map(note => note.amplitude).sort((a, b) => a - b);
  const percentile = index => amplitudes[Math.min(amplitudes.length - 1, Math.max(0, Math.floor(index * (amplitudes.length - 1))))];
  const quiet = percentile(0.2);
  const strong = Math.max(quiet + 0.05, percentile(0.9));

  /** @type {typeof notes[]} */
  let groups = [];
  for (const note of notes) {
    const group = groups.at(-1);
    if (group && note.time - group[0].time <= 0.045) group.push(note);
    else groups.push([note]);
  }

  const localScore = (note, group) => {
    const amplitude = Math.max(0, Math.min(1, (note.amplitude - quiet) / (strong - quiet)));
    const lower = group.filter(candidate => candidate.midi < note.midi).length;
    const upperRank = group.length === 1 ? 0.9 : lower / (group.length - 1);
    const duration = Math.min(1, Math.log1p(note.duration * 3) / Math.log(5.5));
    const registerPenalty = note.midi < 50 ? (50 - note.midi) * 0.08 : 0;
    return amplitude * 0.95 + upperRank * 1.5 + duration * 0.35 - registerPenalty;
  };

  // Basic Pitch intentionally favors recall and may emit weak inner voices.
  // Remove events that are neither locally prominent nor part of the upper
  // voice before solving voice-leading. This lets the path skip accompaniment
  // onsets instead of forcing one selected note at every detected attack.
  groups = groups
    .map(group => group.filter(note => localScore(note, group) >= 1.4))
    .filter(group => group.length > 0);
  if (!groups.length) return [];

  /** @type {{score:number, previous:number}[][]} */
  const states = [];
  for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
    const group = groups[groupIndex];
    const previousGroup = groups[groupIndex - 1];
    states.push(group.map(note => {
      const observation = localScore(note, group);
      if (!previousGroup) return { score: observation, previous: -1 };

      const elapsed = note.time - previousGroup[0].time;
      let best = { score: observation, previous: -1 };
      for (let previousIndex = 0; previousIndex < previousGroup.length; previousIndex++) {
        const previous = previousGroup[previousIndex];
        const interval = Math.abs(note.midi - previous.midi);
        const previousEnd = previous.time + previous.duration;
        const gap = Math.max(0, note.time - previousEnd);
        const overlap = Math.max(0, previousEnd - note.time);
        let continuity;
        if (elapsed > 1.6) continuity = -0.05 * Math.min(interval, 12);
        else if (interval <= 2) continuity = 0.72 - interval * 0.08;
        else if (interval <= 5) continuity = 0.5 - (interval - 2) * 0.08;
        else if (interval <= 12) continuity = 0.2 - (interval - 5) * 0.09;
        else continuity = -0.9 - (interval - 12) * 0.16;
        const transition = continuity - Math.min(0.55, gap * 0.3) - Math.min(0.4, overlap * 0.16);
        const score = states[groupIndex - 1][previousIndex].score + observation + transition;
        if (score > best.score) best = { score, previous: previousIndex };
      }
      return best;
    }));
  }

  let candidateIndex = states.at(-1).reduce((best, state, index, array) => state.score > array[best].score ? index : best, 0);
  const selected = [];
  for (let groupIndex = groups.length - 1; groupIndex >= 0; groupIndex--) {
    selected.push(groups[groupIndex][candidateIndex]);
    const previous = states[groupIndex][candidateIndex].previous;
    if (groupIndex > 0) candidateIndex = previous >= 0 ? previous : states[groupIndex - 1].reduce((best, state, index, array) => state.score > array[best].score ? index : best, 0);
  }
  selected.reverse();

  const voiced = selected.map((note, index) => {
    const next = selected[index + 1];
    const duration = next && next.time < note.time + note.duration - 0.04
      ? Math.max(0.06, next.time - note.time)
      : note.duration;
    return {
      time: Math.round(note.time * 10_000) / 10_000,
      duration: Math.round(duration * 10_000) / 10_000,
      midi: note.midi,
      confidence: Math.round(note.amplitude * 10_000) / 10_000,
      attack: true,
    };
  });

  const registerNormalized = [];
  for (const note of voiced) {
    const previous = registerNormalized.at(-1);
    if (!previous) {
      registerNormalized.push(note);
      continue;
    }
    const interval = Math.abs(note.midi - previous.midi);
    if (note.time - previous.time >= 0.5 || note.duration >= 0.3 || interval < 10) {
      registerNormalized.push(note);
      continue;
    }
    const alternatives = [-24, -12, 0, 12, 24]
      .map(shift => note.midi + shift)
      .filter(midi => midi >= 40 && midi <= 100)
      .sort((a, b) => Math.abs(a - previous.midi) - Math.abs(b - previous.midi));
    registerNormalized.push(alternatives.length && Math.abs(alternatives[0] - previous.midi) <= 4 ? { ...note, midi: alternatives[0] } : note);
  }

  const merged = [];
  for (const note of registerNormalized) {
    const previous = merged.at(-1);
    const previousEnd = previous ? previous.time + previous.duration : 0;
    if (
      previous
      && previous.midi === note.midi
      && note.time - previousEnd <= 0.025
      && (previous.duration < 0.22 || note.duration < 0.22)
    ) {
      previous.duration = Math.round(Math.max(previousEnd, note.time + note.duration) * 10_000 - previous.time * 10_000) / 10_000;
      previous.confidence = Math.round(((previous.confidence + note.confidence) / 2) * 10_000) / 10_000;
    } else {
      merged.push(note);
    }
  }
  return merged;
}

/**
 * Fuse polyphonic note attacks with a perceptual predominant-pitch contour.
 * MELODIA decides which voice is foreground; Basic Pitch supplies the precise
 * attacks that a continuous contour tends to smear or miss on piano.
 *
 * @param {{startTimeSeconds:number,durationSeconds:number,pitchMidi:number,amplitude:number}[]} input
 * @param {{time:number,duration:number,midi:number,confidence:number,attack?:boolean}[]} contourInput
 */
export function fusePredominantMelody(input, contourInput) {
  if (!contourInput?.length) return selectPredominantMelody(input);

  const events = input
    .filter(note => Number.isFinite(note.startTimeSeconds) && Number.isFinite(note.pitchMidi))
    .filter(note => note.durationSeconds >= 0.045 && note.pitchMidi >= 40 && note.pitchMidi <= 100)
    .map(note => ({
      time: Math.max(0, note.startTimeSeconds),
      duration: Math.max(0.06, note.durationSeconds),
      midi: Math.round(note.pitchMidi),
      confidence: Math.max(0, Math.min(1, note.amplitude)),
      attack: true,
    }))
    .sort((a, b) => a.time - b.time || b.confidence - a.confidence);
  const contour = contourInput
    .filter(note => Number.isFinite(note.time) && Number.isFinite(note.midi) && note.duration >= 0.06)
    .map(note => ({ ...note, midi: Math.round(note.midi), attack: true }))
    .sort((a, b) => a.time - b.time);
  if (!events.length || !contour.length) return selectPredominantMelody(input);

  // MELODIA can briefly lock to the same note in a lower harmonic while a
  // piano note decays. Correct a short *run* only when it is bracketed by the
  // original register; genuine lower-register phrases and unbounded leaps are
  // left untouched.
  const stableContour = contour.map(note => ({ ...note }));
  for (let index = 1; index < stableContour.length - 1; index++) {
    const anchor = stableContour[index - 1];
    if (Math.abs(stableContour[index].midi - anchor.midi) < 8) continue;
    for (let endIndex = index + 1; endIndex < stableContour.length; endIndex++) {
      const boundary = stableContour[endIndex];
      if (boundary.time - stableContour[index].time > 1.2) break;
      if (Math.abs(boundary.midi - anchor.midi) > 5) continue;
      const run = stableContour.slice(index, endIndex);
      const shifts = [-12, 12];
      const shift = shifts.reduce((best, candidate) => {
        const shifted = run.map(note => note.midi + candidate);
        if (shifted.some(midi => midi < 40 || midi > 100)) return best;
        const score = Math.abs(shifted[0] - anchor.midi) + Math.abs(shifted.at(-1) - boundary.midi);
        return !best || score < best.score ? { value: candidate, score } : best;
      }, null);
      const originalScore = Math.abs(run[0].midi - anchor.midi) + Math.abs(run.at(-1).midi - boundary.midi);
      if (shift && originalScore - shift.score >= 10) {
        for (let runIndex = index; runIndex < endIndex; runIndex++) stableContour[runIndex].midi += shift.value;
      }
      break;
    }
  }
  for (let index = stableContour.length - 2; index >= 1; index--) {
    const note = stableContour[index];
    const previous = stableContour[index - 1];
    const next = stableContour[index + 1];
    if (
      note.duration <= 0.4
      && next.time - note.time <= 0.8
      && Math.abs(note.midi - previous.midi) >= 10
      && Math.abs(note.midi - next.midi) >= 10
      && Math.abs(next.midi - previous.midi) <= 5
    ) stableContour.splice(index, 1);
  }

  const aligned = [];
  const usedEvents = new Set();
  for (const contourNote of stableContour) {
    const start = contourNote.time - 0.1;
    const end = contourNote.time + contourNote.duration + 0.12;
    const candidates = events.filter((event, index) => (
      !usedEvents.has(index)
      && event.time < end
      && event.time + event.duration > start
      && Math.abs(event.midi - contourNote.midi) <= 2
    ));
    if (!candidates.length) {
      aligned.push({ ...contourNote });
      continue;
    }

    // Preserve separate attacks, but keep only one pitch from simultaneous
    // polyphony by choosing the note closest to the perceptual contour.
    const onsetGroups = [];
    for (const candidate of candidates) {
      const group = onsetGroups.at(-1);
      if (group && candidate.time - group[0].time <= 0.045) group.push(candidate);
      else onsetGroups.push([candidate]);
    }
    for (const group of onsetGroups) {
      const selected = group.reduce((best, candidate) => {
        const candidateScore = Math.abs(candidate.midi - contourNote.midi) * 2 - candidate.confidence;
        const bestScore = Math.abs(best.midi - contourNote.midi) * 2 - best.confidence;
        return candidateScore < bestScore ? candidate : best;
      });
      usedEvents.add(events.indexOf(selected));
      aligned.push({ ...selected });
    }
  }

  // Continuous-contour methods can miss the first piano attacks or brief gaps.
  // Fill only near an adjacent contour register; do not invent a voice in a
  // long unvoiced section or follow an unrelated bass entry.
  const fallback = selectPredominantMelody(input);
  for (const note of fallback) {
    if (aligned.some(item => Math.abs(item.time - note.time) <= 0.045)) continue;
    const previous = [...stableContour].reverse().find(item => item.time <= note.time);
    const next = stableContour.find(item => item.time > note.time);
    const coveringContour = stableContour.find(item => (
      note.time >= item.time - 0.1 && note.time <= item.time + item.duration + 0.12
    ));
    // A segmented contour deliberately smooths over short piano ornaments.
    // Reintroduce only the voice-leading winner, within a musically local
    // interval of the perceived contour.
    if (coveringContour) {
      if (Math.abs(note.midi - coveringContour.midi) <= 7) aligned.push({ ...note });
      continue;
    }
    const previousGap = previous ? note.time - (previous.time + previous.duration) : Infinity;
    const nextGap = next ? next.time - (note.time + note.duration) : Infinity;
    const nearBoundary = Math.min(previousGap, nextGap) <= 2.5;
    const previousDistance = previous ? Math.abs(note.midi - previous.midi) : Infinity;
    const nextDistance = next ? Math.abs(note.midi - next.midi) : Infinity;
    const closeToAvailableBoundary = previous && next
      ? previousDistance <= 7 && nextDistance <= 7
      : Math.min(previousDistance, nextDistance) <= 7;
    if (nearBoundary && closeToAvailableBoundary) aligned.push({ ...note });
  }

  aligned.sort((a, b) => a.time - b.time || b.confidence - a.confidence);
  const result = [];
  for (const note of aligned) {
    const previous = result.at(-1);
    if (previous && Math.abs(note.time - previous.time) <= 0.045) {
      if (note.confidence > previous.confidence) result[result.length - 1] = note;
      continue;
    }
    if (previous && previous.midi === note.midi && note.time - (previous.time + previous.duration) <= 0.035) {
      previous.duration = Math.round((note.time + note.duration - previous.time) * 10_000) / 10_000;
      previous.confidence = Math.round(((previous.confidence + note.confidence) / 2) * 10_000) / 10_000;
      continue;
    }
    result.push({
      ...note,
      time: Math.round(note.time * 10_000) / 10_000,
      duration: Math.round(note.duration * 10_000) / 10_000,
      confidence: Math.round(note.confidence * 10_000) / 10_000,
      attack: true,
    });
  }
  // Piano note durations include pedal/resonance. A karaoke pitch bar is
  // monophonic, so end each bar at the following detected attack.
  return result.map((note, index) => {
    const next = result[index + 1];
    if (!next || next.time >= note.time + note.duration) return note;
    return {
      ...note,
      duration: Math.round(Math.max(0.06, next.time - note.time) * 10_000) / 10_000,
    };
  });
}
