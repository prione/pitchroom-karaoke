import assert from "node:assert/strict";
import test from "node:test";
import { getMicAudioConstraints, getMicGateSettings } from "../app/mic-settings.mjs";

test("higher mic sensitivity lowers the voice gate without removing it", () => {
  const low = getMicGateSettings(0);
  const normal = getMicGateSettings(50);
  const high = getMicGateSettings(100);

  assert.ok(low.minimumRms > normal.minimumRms);
  assert.ok(normal.minimumRms > high.minimumRms);
  assert.ok(low.noiseMultiplier > normal.noiseMultiplier);
  assert.ok(normal.noiseMultiplier > high.noiseMultiplier);
  assert.ok(low.minimumClarity > normal.minimumClarity);
  assert.ok(normal.minimumClarity > high.minimumClarity);
  assert.ok(high.minimumRms > 0);
});

test("mic constraints target a selected input only when one is selected", () => {
  assert.equal("deviceId" in getMicAudioConstraints(), false);
  assert.deepEqual(getMicAudioConstraints("microphone-2").deviceId, { exact: "microphone-2" });
});
