// Shared helpers for both dashboard pages.
//
// Everything the model or a participant produced is inserted with textContent,
// never innerHTML. That, plus the Content-Security-Policy the server sends, is
// why a transcript containing markup cannot become markup in this page.

export const CSRF_HEADER = 'X-Requested-With';
export const CSRF_VALUE = 'teams-notes';

export async function api(path, options = {}) {
  const method = (options.method || 'GET').toUpperCase();
  const headers = { Accept: 'application/json' };
  if (method !== 'GET') {
    headers[CSRF_HEADER] = CSRF_VALUE;
    if (options.body !== undefined) headers['Content-Type'] = 'application/json';
  }
  const response = await fetch(path, {
    method,
    headers,
    credentials: 'same-origin',
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });

  let payload = null;
  const text = await response.text();
  if (text) {
    try { payload = JSON.parse(text); } catch (err) { payload = { error: text }; }
  }

  if (!response.ok) {
    const message = (payload && (payload.error || payload.detail)) || `Request failed (${response.status})`;
    const remedy = (payload && payload.remedy) || '';
    const error = new Error(remedy ? `${message} ${remedy}` : message);
    error.status = response.status;
    throw error;
  }
  return payload;
}

export function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  if (options.href) node.href = options.href;
  if (options.title) node.title = options.title;
  if (options.type) node.type = options.type;
  if (options.value !== undefined) node.value = options.value;
  if (options.checked !== undefined) node.checked = options.checked;
  if (options.dataset) {
    Object.entries(options.dataset).forEach(([key, value]) => { node.dataset[key] = value; });
  }
  if (options.onClick) node.addEventListener('click', options.onClick);
  if (options.onChange) node.addEventListener('change', options.onChange);
  children.filter(Boolean).forEach((child) => node.appendChild(child));
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return hours ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`;
}

export function formatDateTime(isoString) {
  if (!isoString) return '';
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return String(isoString);
  return date.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

export function showError(container, message) {
  clear(container);
  container.classList.remove('hidden');
  container.appendChild(el('strong', { text: 'Something went wrong. ' }));
  container.appendChild(document.createTextNode(String(message)));
}

export function hide(node) { node.classList.add('hidden'); }
export function show(node) { node.classList.remove('hidden'); }

export async function copyText(text, button) {
  const original = button.textContent;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = 'Copied';
  } catch (err) {
    // Clipboard access can be blocked; fall back to selecting the text.
    button.textContent = 'Select it manually';
  }
  setTimeout(() => { button.textContent = original; }, 1800);
}
