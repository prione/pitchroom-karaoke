/**
 * Keep the guide readable without erasing intentional fast ornaments.
 *
 * @param {{time:number, duration?:number, midi:number, confidence:number, attack?:boolean}[]} input
 * @param {boolean} preserveOrnaments
 */
export function simplifyPitchBars(input, preserveOrnaments = true) {
  const mergeMatchingBars = (bars, maxGap) => {
    const merged = [];
    for (const source of bars) {
      const bar = { ...source, duration: Math.max(0.06, source.duration ?? 0.1) };
      const previous = merged.at(-1);
      const previousEnd = previous ? previous.time + (previous.duration ?? 0.1) : 0;
      if (previous && !bar.attack && Math.abs(previous.midi - bar.midi) < 0.5 && bar.time - previousEnd <= maxGap) {
        previous.duration = Math.max(previous.duration ?? 0.1, bar.time + (bar.duration ?? 0.1) - previous.time);
        previous.confidence = (previous.confidence + bar.confidence) / 2;
      } else {
        merged.push(bar);
      }
    }
    return merged;
  };

  const bars = mergeMatchingBars([...input].sort((a, b) => a.time - b.time), 0.07);
  const simplified = [];
  for (let index = 0; index < bars.length; index++) {
    const bar = bars[index];
    const duration = bar.duration ?? 0.1;
    const previous = simplified.at(-1);
    const next = bars[index + 1];
    if (duration < 0.15) {
      const previousGap = previous ? bar.time - (previous.time + (previous.duration ?? 0.1)) : Infinity;
      const nextGap = next ? next.time - (bar.time + duration) : Infinity;
      const returnsToPrevious = previous && next && previousGap <= 0.07 && nextGap <= 0.07 && Math.abs(previous.midi - next.midi) < 0.5;
      const turnInterval = previous ? Math.abs(previous.midi - bar.midi) : 0;
      const isOrnamentTurn = returnsToPrevious && turnInterval >= 0.75 && turnInterval <= 5;

      // A short A-B-A turn is the basic unit of a trill or mordent. Keep it,
      // while still collapsing implausible large single-frame pitch glitches.
      if (returnsToPrevious && (!preserveOrnaments || !isOrnamentTurn)) {
        previous.duration = next.time + (next.duration ?? 0.1) - previous.time;
        previous.confidence = (previous.confidence + bar.confidence + next.confidence) / 3;
        index += 1;
        continue;
      }
      if (previous && previousGap <= 0.045 && Math.abs(previous.midi - bar.midi) < 0.5) {
        previous.duration = bar.time + duration - previous.time;
        previous.confidence = (previous.confidence + bar.confidence) / 2;
        continue;
      }
      if (next && nextGap <= 0.045 && Math.abs(next.midi - bar.midi) < 0.5) {
        next.duration = next.time + (next.duration ?? 0.1) - bar.time;
        next.time = bar.time;
        next.confidence = (next.confidence + bar.confidence) / 2;
        continue;
      }
    }
    simplified.push({ ...bar, midi: Math.round(bar.midi) });
  }
  return mergeMatchingBars(simplified, 0.08);
}
