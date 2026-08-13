/**
 * Immediately detach every audio source belonging to the previous song.
 * React state updates alone are not enough here: a pending play() promise can
 * otherwise resume an element before the next render unmounts it.
 *
 * @param {{pause:Function,removeAttribute:Function,load:Function}|null} audio
 * @param {{pause:Function,removeAttribute:Function,load:Function}|null} guide
 * @param {{pause:Function}|null} youtube
 */
export function detachMediaSession(audio, guide, youtube) {
  for (const element of [audio, guide]) {
    if (!element) continue;
    element.pause();
    element.removeAttribute("src");
    element.load();
  }
  youtube?.pause();
}

export function isCurrentPlayback(command, activeCommand, audio, currentAudio) {
  return command === activeCommand && audio === currentAudio;
}
