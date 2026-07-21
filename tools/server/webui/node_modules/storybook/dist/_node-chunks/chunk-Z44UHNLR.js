import CJS_COMPAT_NODE_URL_sru4mlhvd6 from 'node:url';
import CJS_COMPAT_NODE_PATH_sru4mlhvd6 from 'node:path';
import CJS_COMPAT_NODE_MODULE_sru4mlhvd6 from "node:module";

var __filename = CJS_COMPAT_NODE_URL_sru4mlhvd6.fileURLToPath(import.meta.url);
var __dirname = CJS_COMPAT_NODE_PATH_sru4mlhvd6.dirname(__filename);
var require = CJS_COMPAT_NODE_MODULE_sru4mlhvd6.createRequire(import.meta.url);

// ------------------------------------------------------------
// end of CJS compatibility banner, injected by Storybook's esbuild configuration
// ------------------------------------------------------------
import {
  any
} from "./chunk-2FN7EC43.js";

// src/cli/detect.ts
async function detectPnp() {
  return !!any([".pnp.js", ".pnp.cjs"]);
}

export {
  detectPnp
};
