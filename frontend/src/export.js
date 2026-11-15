// Category predictions are independent of whether Qwen produced draft text.
export function resultsCsv(items) {
  const cell = value => `"${String(value).replaceAll('"', '""')}"`;
  const hasErrors = items.some(item => item.incomplete || item.error || item.errors?.length);
  const rows = items.map(item => {
    const ids = item.microcats === undefined ? item.drafts.map(d => d.mcId) : item.microcats;
    const texts = ids === null ? null : ids.map(id => item.drafts.find(d => d.mcId === id)?.text ?? null);
    return [item.testcase ?? `case-${String(item.index + 1).padStart(3,'0')}`,
      JSON.stringify(ids), JSON.stringify(texts),
      ...(hasErrors ? [JSON.stringify(item.error ?? item.errors ?? (item.incomplete ? { message:'Обработка не завершена' } : null))] : []),
    ].map(cell).join(',');
  });
  return '\uFEFFtestcase,microcats,descriptions' + (hasErrors ? ',error' : '') + '\r\n' + rows.join('\r\n') + '\r\n';
}
