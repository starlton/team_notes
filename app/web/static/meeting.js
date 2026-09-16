import { api, el, clear, formatDuration, formatDateTime, showError, hide, show, copyText } from '/static/common.js';

const match = window.location.pathname.match(/\/meeting\/(\d+)/);
const meetingId = match ? Number(match[1]) : 0;

const dom = {
  title: document.getElementById('meeting-title'),
  meta: document.getElementById('meeting-meta'),
  errorBox: document.getElementById('error-box'),
  warningsBox: document.getElementById('warnings-box'),
  progressBox: document.getElementById('progress-box'),
  progressStage: document.getElementById('progress-stage'),
  progressFill: document.getElementById('progress-fill'),
  informed: document.getElementById('informed-toggle'),
  summaryPanel: document.getElementById('summary-panel'),
  summaryText: document.getElementById('summary-text'),
  summaryBullets: document.getElementById('summary-bullets'),
  prioritiesPanel: document.getElementById('priorities-panel'),
  prioritiesList: document.getElementById('priorities-list'),
  actionsPanel: document.getElementById('actions-panel'),
  actionsList: document.getElementById('actions-list'),
  speakersPanel: document.getElementById('speakers-panel'),
  speakersList: document.getElementById('speakers-list'),
  saveNames: document.getElementById('save-names-btn'),
  saveRegen: document.getElementById('save-regen-btn'),
  draftsPanel: document.getElementById('drafts-panel'),
  draftsList: document.getElementById('drafts-list'),
  transcriptList: document.getElementById('transcript-list'),
  noTranscript: document.getElementById('no-transcript'),
  download: document.getElementById('download-transcript'),
  audio: document.getElementById('audio-player'),
  regenerate: document.getElementById('regenerate-btn'),
  deleteBtn: document.getElementById('delete-btn'),
};

let pollTimer = null;

function reportError(err) { showError(dom.errorBox, err.message || err); }
function clearError() { hide(dom.errorBox); clear(dom.errorBox); }

// --- renderers --------------------------------------------------------------

function renderHeader(meeting) {
  document.title = `${meeting.title} — Teams Notes`;
  dom.title.textContent = meeting.title;
  const parts = [formatDateTime(meeting.started_at), formatDuration(meeting.duration_seconds)];
  if (meeting.source === 'auto') parts.push('detected automatically');
  dom.meta.textContent = parts.filter(Boolean).join(' · ');
  dom.informed.checked = Boolean(meeting.participants_informed);
  dom.download.href = `/api/meetings/${meetingId}/transcript.txt`;

  if (meeting.has_audio) {
    dom.audio.src = `/api/meetings/${meetingId}/audio`;
    show(dom.audio);
  } else {
    hide(dom.audio);
  }

  const warnings = meeting.warnings || [];
  clear(dom.warningsBox);
  if (warnings.length || meeting.error) {
    show(dom.warningsBox);
    dom.warningsBox.classList.toggle('error', Boolean(meeting.error));
    dom.warningsBox.appendChild(el('strong', { text: meeting.error ? 'This meeting did not finish processing' : 'Notes' }));
    const list = el('ul', { className: 'clean' });
    if (meeting.error) list.appendChild(el('li', { text: meeting.error }));
    warnings.forEach((item) => list.appendChild(el('li', { text: String(item) })));
    dom.warningsBox.appendChild(list);
  } else {
    hide(dom.warningsBox);
  }

  if (meeting.status === 'processing' || meeting.status === 'recorded') {
    show(dom.progressBox);
    dom.progressStage.textContent = meeting.stage || 'Waiting to start…';
    dom.progressFill.style.width = `${Math.round((meeting.progress || 0) * 100)}%`;
  } else {
    hide(dom.progressBox);
  }
}

function renderSummary(summary) {
  const hasContent = Boolean(summary.summary || (summary.bullets || []).length);
  if (!hasContent) { hide(dom.summaryPanel); return; }
  show(dom.summaryPanel);
  dom.summaryText.textContent = summary.summary || '';
  clear(dom.summaryBullets);
  (summary.bullets || []).forEach((bullet) => {
    dom.summaryBullets.appendChild(el('li', { text: String(bullet) }));
  });
}

function renderPriorities(priorities) {
  if (!priorities.length) { hide(dom.prioritiesPanel); return; }
  show(dom.prioritiesPanel);
  clear(dom.prioritiesList);
  priorities.forEach((item) => {
    dom.prioritiesList.appendChild(el('div', { className: 'item' }, [
      el('div', { className: 'row' }, [
        el('span', { className: `tag ${item.priority}`, text: item.priority }),
        el('strong', { text: item.point }),
      ]),
      item.reason ? el('p', { className: 'small muted', text: item.reason }) : null,
    ]));
  });
}

function renderActions(actions) {
  if (!actions.length) { hide(dom.actionsPanel); return; }
  show(dom.actionsPanel);
  clear(dom.actionsList);
  actions.forEach((item) => {
    const row = el('div', { className: `item${item.done ? ' done' : ''}` });
    const checkbox = el('input', { type: 'checkbox', checked: Boolean(item.done) });
    checkbox.addEventListener('change', async () => {
      try {
        await api(`/api/meetings/${meetingId}/actions/${item.id}`, {
          method: 'POST', body: { done: checkbox.checked },
        });
        row.classList.toggle('done', checkbox.checked);
      } catch (err) {
        checkbox.checked = !checkbox.checked;
        reportError(err);
      }
    });

    const meta = [item.owner, item.due, item.priority].filter(Boolean).join(' · ');
    row.appendChild(el('label', { className: 'row' }, [
      checkbox,
      el('span', { className: 'item-task', text: item.task }),
      el('span', { className: `tag ${item.priority}`, text: meta }),
    ]));
    if (item.context) row.appendChild(el('p', { className: 'small muted', text: item.context }));
    dom.actionsList.appendChild(row);
  });
}

function renderSpeakers(speakers) {
  if (!speakers.length) { hide(dom.speakersPanel); return; }
  show(dom.speakersPanel);
  clear(dom.speakersList);
  speakers.forEach((speaker) => {
    const input = el('input', { type: 'text', value: speaker.display_name || speaker.label });
    input.dataset.label = speaker.label;
    input.maxLength = 120;

    const share = Math.round((speaker.share || 0) * 100);
    const fill = el('div');
    fill.style.width = `${share}%`;

    const points = el('ul', { className: 'clean small' });
    (speaker.key_points || []).forEach((point) => {
      points.appendChild(el('li', { text: String(point) }));
    });

    dom.speakersList.appendChild(el('div', { className: 'item' }, [
      el('div', { className: 'row' }, [
        input,
        el('span', { className: 'bar-track' }, [fill]),
        el('span', { className: 'small muted',
          text: `${share}% · ${formatDuration(speaker.talk_seconds)} · ${speaker.word_count} words` }),
      ]),
      speaker.summary ? el('p', { className: 'small', text: speaker.summary }) : null,
      (speaker.key_points || []).length ? points : null,
    ]));
  });
}

function renderDrafts(drafts) {
  if (!drafts.length) { hide(dom.draftsPanel); return; }
  show(dom.draftsPanel);
  clear(dom.draftsList);
  drafts.forEach((draft) => {
    const copyButton = el('button', { text: 'Copy' });
    copyButton.addEventListener('click', () => {
      const text = draft.subject ? `Subject: ${draft.subject}\n\n${draft.body}` : draft.body;
      copyText(text, copyButton);
    });
    dom.draftsList.appendChild(el('div', { className: 'item' }, [
      el('div', { className: 'row' }, [
        el('span', { className: 'tag', text: draft.kind }),
        el('strong', { text: draft.subject || draft.audience || 'Follow-up' }),
        el('span', { className: 'spacer' }),
        copyButton,
      ]),
      draft.audience ? el('p', { className: 'small muted', text: `To: ${draft.audience}` }) : null,
      el('pre', { className: 'draft-body', text: draft.body }),
    ]));
  });
}

function renderTranscript(lines) {
  clear(dom.transcriptList);
  if (!lines.length) { show(dom.noTranscript); return; }
  hide(dom.noTranscript);
  lines.forEach((line) => {
    dom.transcriptList.appendChild(el('div', { className: 'line' }, [
      el('span', { className: 'ts', text: line.timestamp }),
      el('span', { className: 'who', text: line.speaker }),
      el('span', { className: 'what', text: line.text }),
    ]));
  });
}

// --- loading ----------------------------------------------------------------

async function load() {
  try {
    const data = await api(`/api/meetings/${meetingId}`);
    renderHeader(data.meeting);
    renderSummary(data.summary || {});
    renderPriorities(data.priorities || []);
    renderActions(data.action_items || []);
    renderSpeakers(data.speakers || []);
    renderDrafts(data.drafts || []);
    renderTranscript(data.transcript || []);
    clearError();

    const inFlight = data.meeting.status === 'processing' || data.meeting.status === 'recorded';
    if (inFlight && pollTimer === null) pollTimer = setInterval(load, 2500);
    if (!inFlight && pollTimer !== null) { clearInterval(pollTimer); pollTimer = null; }
  } catch (err) {
    reportError(err);
  }
}

async function saveNames(regenerate) {
  const names = {};
  dom.speakersList.querySelectorAll('input[type="text"]').forEach((input) => {
    names[input.dataset.label] = input.value.trim();
  });
  try {
    await api(`/api/meetings/${meetingId}/speakers`, {
      method: 'POST', body: { names, regenerate },
    });
    clearError();
    await load();
  } catch (err) {
    reportError(err);
  }
}

dom.saveNames.addEventListener('click', () => saveNames(false));
dom.saveRegen.addEventListener('click', () => saveNames(true));

dom.informed.addEventListener('change', async () => {
  try {
    await api(`/api/meetings/${meetingId}`, {
      method: 'PATCH', body: { participants_informed: dom.informed.checked },
    });
  } catch (err) {
    dom.informed.checked = !dom.informed.checked;
    reportError(err);
  }
});

dom.regenerate.addEventListener('click', async () => {
  try {
    await api(`/api/meetings/${meetingId}/regenerate`, { method: 'POST' });
    await load();
  } catch (err) {
    reportError(err);
  }
});

dom.deleteBtn.addEventListener('click', async () => {
  if (!window.confirm('Delete this meeting, its notes and its audio? This cannot be undone.')) return;
  try {
    await api(`/api/meetings/${meetingId}`, { method: 'DELETE' });
    window.location.href = '/';
  } catch (err) {
    reportError(err);
  }
});

if (!meetingId) {
  reportError(new Error('That meeting link is not valid.'));
} else {
  load();
}
