import { mkdir, copyFile, cp } from 'node:fs/promises';
const root = new URL('../', import.meta.url);
const dist = new URL('dist/', root);
await mkdir(dist, { recursive: true });
await Promise.all([
  copyFile(new URL('index.html', root), new URL('index.html', dist)),
  copyFile(new URL('favicon.svg', root), new URL('favicon.svg', dist)),
  cp(new URL('src/', root), new URL('src/', dist), { recursive: true }),
]);
console.log('Готово: frontend/dist');
