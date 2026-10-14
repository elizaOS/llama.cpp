/** Removes intermediate WebUI assets after the standalone HTML bundle is built. */
import { rmSync } from 'node:fs';

const publicDir = new URL('../../public/', import.meta.url);
rmSync(new URL('_app', publicDir), { recursive: true, force: true });
rmSync(new URL('favicon.svg', publicDir), { force: true });
rmSync(new URL('index.html.gz', publicDir), { force: true });
