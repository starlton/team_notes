import { api, el, clear, formatDuration, formatDateTime, showError, hide, show } from '/static/common.js';

const dom = {
  statusDot: document.getElementById('status-dot'),
  statusText: document.getElementById('status-text'),
  errorBox: document.getElementById('error-box'),
  consentNotice: document.getElementById('consent-notice'),
  consentText: document.getElementById('consent-text'),
  consentAccept: document.getElementById('consent-accept'),
  title: document.getElementById('meeting-title'),
  startBtn: document.getElementById('start-btn'),
  stopBtn: document.getElementById('stop-btn'),
  discardBtn: document.getElementById('discard-btn'),
  levelsRow: document.getElementById('levels-row'),
  informed: document.getElementById('informed-toggle'),
  reminder: document.getElementById('reminder-toggle'),
  autoDetect: document.getElementById('autodetect-toggle'),
  healthGrid: document.getElementById('health-grid'),
  meetingList: document.getElementById('meeting-list'),
  noMeetings: document.getElementById('no-meetings'),
};

let consentAcknowledged = false;
let busy = false;

function reportError(err) {
  showError(dom.errorBox, err.message || err);
}

function clearError() {
  hide(dom.errorBox);
  clear(dom.errorBox);
}

// --- status -----------------------------------------------------------------

function renderLevels(levels) {
  clear(dom.levelsRow);
  const entries = Object.entries(levels || {});
  if (!entries.length) return;
  entries.forEach(([name, dbfs]) => {
    // -60 dBFS is effectively silence, 0 is full scale.
    const ratio = dbfs === null ? 0 : Math.max(0, Math.min(1, (Number(dbfs) + 60) / 60));
    const fill = el('div');
    fill.style.width = `${Math.round(ratio * 100)}%`;
    dom.levelsRow.appendChild(el('span', { text: `${name}:` }));
    dom.levelsRow.appendChild(el('span', { className: 'bar-track' }, [fill]));
  });
}

function renderStatus(status) {
  const recording = Boolean(status.recording);
  dom.statusDot.className = recording ? 'dot live' : 'dot';
  const jobs = status.jobs || {};
  let text;
  if (recording) {
    text = `Recording — ${formatDuration(status.elapsed_seconds)}`;
  } else if (jobs.running) {
    text = jobs.running.description;
  } else if ((jobs.queued || []).length) {
    text = `${jobs.queued.length} meeting(s) queued`;
  } else {
    text = 'Idle';
  }
  dom.statusText.textContent = text;

  dom.startBtn.disabled = recording || busy || !consentAcknowledged;
  dom.stopBtn.disabled = !recording || busy;
  dom.discardBtn.disabled = !recording || busy;
  dom.title.disabled = recording;
  renderLevels(status.levels);

  const consent = status.consent || {};
  consentAcknowledged = Boolean(consent.acknowledged);
  dom.reminder.checked = Boolean(consent.reminder_enabled);
  dom.autoDetect.checked = Boolean(status.auto_detect_enabled);
  dom.consentText.textContent = consent.notice || '';
  if (consentAcknowledged) hide(dom.consentNotice); else show(dom.consentNotice);
}

// --- health -----------------------------------------------------------------

function healthCard(name, ok, detail, remedy) {
  return el('div', { className: 'panel', title: remedy || '' }, [
    el('div', { className: 'row' }, [
      el('span', { className: ok ? 'dot ok' : 'dot bad' }),
      el('strong', { text: name }),
    ]),
    el('p', { className: 'small muted', text: detail }),
    remedy ? el('p', { className: 'small', text: remedy }) : null,
  ]);
}

function renderHealth(health) {
  clear(dom.healthGrid);
  const ollama = health.ollama || {};
  dom.healthGrid.appendChild(healthCard(
    'Local model (Ollama)', ollama.ok,
    ollama.ok ? `Ready: ${ollama.model}` : (ollama.error || 'Unavailable'),
    ollama.ok ? '' : (ollama.remedy || '')));

  const diarization = health.diarization || {};
  dom.healthGrid.appendChild(healthCard(
    'Speaker labels', diarization.ok,
    diarization.ok ? 'Hugging Face token found' : (diarization.error || ''),
    diarization.ok ? '' : (diarization.remedy || '')));

  const whisper = health.whisper || {};
  dom.healthGrid.appendChild(healthCard(
    'Transcription', true,
    `${whisper.model || 'small.en'} (${whisper.compute_type || 'int8'})`, ''));

  const storage = health.storage || {};
  dom.healthGrid.appendChild(healthCard(
    'Storage', true,
    `${storage.meetings || 0} meeting(s) stored locally`, storage.data_dir || ''));
}

// --- meetings ---------------------------------------------------------------

function statusTag(meeting) {
  const label = {
    complete: 'Ready', processing: 'Processing…', recorded: 'Waiting to process',
    recording: 'Recording', failed: 'Failed', discarded: 'Discarded',
  }[meeting.status] || meeting.status;
  return el('span', { className: `tag ${meeting.status}`, text: label });
}

function meetingRow(meeting) {
  const header = el('div', { className: 'row' }, [
    el('a', { className: 'meeting-title', text: meeting.title, href: `/meeting/${meeting.id}` }),
    statusTag(meeting),
    meeting.source === 'auto' ? el('span', { className: 'tag', text: 'auto' }) : null,
    meeting.participants_informed ? el('span', { className: 'tag', text: 'participants informed' }) : null,
  ]);

  const meta = el('div', { className: 'small muted',
    text: `${formatDateTime(meeting.started_at)} · ${meeting.duration_label}` });

  const children = [header, meta];

  if (meeting.summary) {
    children.push(el('p', { className: 'small', text: meeting.summary }));
  }
  if (meeting.status === 'processing') {
    const fill = el('div');
    fill.style.width = `${Math.round((meeting.progress || 0) * 100)}%`;
    children.push(el('div', { className: 'small muted', text: meeting.stage || 'Working…' }));
    children.push(el('div', { className: 'progress' }, [fill]));
  }
  if (meeting.status === 'failed' && meeting.error) {
    children.push(el('p', { className: 'small', text: meeting.error }));
  }
  return el('li', {}, children);
}

async function refreshMeetings() {
  const data = await api('/api/meetings?limit=50');
  clear(dom.meetingList);
  (data.meetings || []).forEach((meeting) => dom.meetingList.appendChild(meetingRow(meeting)));
  if (!(data.meetings || []).length) show(dom.noMeetings); else hide(dom.noMeetings);
}

// --- actions ----------------------------------------------------------------

async function withBusy(fn) {
  busy = true;
  try {
    clearError();
    await fn();
  } catch (err) {
    reportError(err);
  } finally {
    busy = false;
    await refresh();
  }
}

dom.consentAccept.addEventListener('click', () => withBusy(async () => {
  await api('/api/consent', { method: 'POST', body: { acknowledged: true } });
}));

dom.startBtn.addEventListener('click', () => withBusy(async () => {
  await api('/api/recording/start', {
    method: 'POST',
    body: { title: dom.title.value.trim(), participants_informed: dom.informed.checked },
  });
  dom.title.value = '';
}));

dom.stopBtn.addEventListener('click', () => withBusy(async () => {
  await api('/api/recording/stop', { method: 'POST' });
}));

dom.discardBtn.addEventListener('click', () => withBusy(async () => {
  if (!window.confirm('Discard this recording and delete its audio?')) return;
  await api('/api/recording/discard', { method: 'POST' });
}));

dom.reminder.addEventListener('change', () => withBusy(async () => {
  await api('/api/consent', { method: 'POST', body: { reminder_enabled: dom.reminder.checked } });
}));

dom.autoDetect.addEventListener('change', () => withBusy(async () => {
  await api('/api/auto-detect', { method: 'POST', body: { enabled: dom.autoDetect.checked } });
}));

// --- polling ----------------------------------------------------------------

async function refresh() {
  try {
    renderStatus(await api('/api/status'));
    await refreshMeetings();
    clearError();
  } catch (err) {
    reportError(err);
  }
}

async function refreshHealth() {
  try {
    renderHealth(await api('/api/health'));
  } catch (err) {
    reportError(err);
  }
}

refresh();
refreshHealth();
setInterval(refresh, 2000);
setInterval(refreshHealth, 30000);
