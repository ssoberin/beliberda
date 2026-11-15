// Send original input. Only the backend determines format and creates records.
export function prepareSubmission(value, title = '', file = null) {
  if (!file && !value.trim()) throw new Error('Введите описание услуги или загрузите файл.');
  const payload = new FormData();
  if (file) payload.append('file', file);
  else payload.append('text', value);
  payload.append('title', title);
  return { payload };
}
