/** 转义 HTML，再对法律关键字加粗渲染 */
export function formatMessage(text: string): string {
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')

  // 模型输出的 Markdown 加粗
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  // 法律名称
  html = html.replace(/《([^》]+)》/g, '<strong>《$1》</strong>')
  // 法条号
  html = html.replace(/(第[零〇一二三四五六七八九十百千万\d]+条)/g, '<strong>$1</strong>')
  // 章节标题
  html = html.replace(/(【[^】]+】)/g, '<strong class="section-title">$1</strong>')

  return html.replace(/\n/g, '<br>')
}
