import { mkdir, copyFile, cp, rm } from 'node:fs/promises';
const root = new URL('../', import.meta.url);
const dist = new URL('dist/', root);
await mkdir(dist, { recursive: true });
await rm(new URL('favicon.svg', dist), { force: true });
await Promise.all([
  copyFile(new URL('index.html', root), new URL('index.html', dist)),
  cp(new URL('src/', root), new URL('src/', dist), { recursive: true }),
]);
console.log('Готово: frontend/dist');
