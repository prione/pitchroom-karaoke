import assert from "node:assert/strict";
import test from "node:test";

import { detachMediaSession, isCurrentPlayback } from "../app/media-session.mjs";

function fakeMedia() {
  const calls = [];
  return {
    calls,
    pause: () => calls.push("pause"),
    removeAttribute: name => calls.push(`remove:${name}`),
    load: () => calls.push("load"),
  };
}

test("detaches both source and guide before switching songs", () => {
  const source = fakeMedia();
  const guide = fakeMedia();
  let youtubePaused = false;

  detachMediaSession(source, guide, { pause: () => { youtubePaused = true; } });

  assert.deepEqual(source.calls, ["pause", "remove:src", "load"]);
  assert.deepEqual(guide.calls, ["pause", "remove:src", "load"]);
  assert.equal(youtubePaused, true);
});

test("rejects a play completion from an obsolete song session", () => {
  const oldAudio = {};
  const newAudio = {};

  assert.equal(isCurrentPlayback(4, 5, oldAudio, newAudio), false);
  assert.equal(isCurrentPlayback(5, 5, oldAudio, newAudio), false);
  assert.equal(isCurrentPlayback(5, 5, newAudio, newAudio), true);
});
