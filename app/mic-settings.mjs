export function getMicGateSettings(sensitivity) {
  const normalized = Math.max(0, Math.min(100, Number(sensitivity) || 0)) / 100;
  return {
    minimumRms: 0.018 - 0.015 * normalized,
    noiseMultiplier: 3.4 - 1.9 * normalized,
    minimumClarity: 0.68 - 0.12 * normalized,
  };
}

export function getMicAudioConstraints(deviceId = "") {
  return {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: false,
    channelCount: 1,
    ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
  };
}
