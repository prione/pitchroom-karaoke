const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("pitchroomDesktop", Object.freeze({
  isElectron: true,
  platform: process.platform,
  versions: Object.freeze({
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  }),
}));

