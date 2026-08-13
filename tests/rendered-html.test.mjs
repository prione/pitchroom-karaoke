import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("builds the PitchRoom application shell", async () => {
  const html = await readFile(new URL("../dist-renderer/index.html", import.meta.url), "utf8");
  assert.match(html, /<title>PITCHROOM/);
  assert.match(html, /<div id="root"><\/div>/);
  assert.match(html, /assets\/index-/);
});

test("starts without a preloaded sample song", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /useState\("曲を選択してください"\)/);
  assert.match(page, /useState<SourceKind>\("none"\)/);
  assert.doesNotMatch(page, /sample\.(mp3|wav|m4a)/i);
});
