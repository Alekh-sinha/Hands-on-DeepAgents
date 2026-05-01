/* Skills Agent — main app */
'use strict';

// ── Auth ───────────────────────────────────────────────────────────────────
const token = localStorage.getItem('sa_token');
const email = localStorage.getItem('sa_email') || '';

if (!token) { window.location.href = '/'; }

document.getElementById('user-email').textContent = email;

// ── API helpers ────────────────────────────────────────────────────────────
const api = {
  headers() {
    return { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` };
  },
  async get(path) {
    const r = await fetch(path, { headers: this.headers() });
    if (r.status === 401) { logout(); return null; }
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, { method: 'POST', headers: this.headers(), body: JSON.stringify(body) });
    if (r.status === 401) { logout(); return null; }
    return r.json();
  },
  async put(path) {
    const r = await fetch(path, { method: 'PUT', headers: this.headers() });
    return r.json();
  },
  async del(path) {
    const r = await fetch(path, { method: 'DELETE', headers: this.headers() });
    return r.json();
  },
};

// ── Logout ─────────────────────────────────────────────────────────────────
async function logout() {
  await fetch('/auth/logout', { method: 'POST', headers: { Authorization: `Bearer ${token}` } })
    .catch(() => {});
  localStorage.removeItem('sa_token');
  localStorage.removeItem('sa_email');
  window.location.href = '/';
}
document.getElementById('logout-btn').addEventListener('click', logout);

// ── Tabs ───────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const name = btn.dataset.tab;
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`tab-${name}`).classList.add('active');
  });
});

// ── Skills ─────────────────────────────────────────────────────────────────
const skillsList  = document.getElementById('skills-list');
const skillsEmpty = document.getElementById('skills-empty');

async function loadSkills() {
  const data = await api.get('/api/skills');
  if (!data) return;
  renderSkills(data.skills || []);
}

function renderSkills(skills) {
  // Remove all tile children but keep the empty state node
  Array.from(skillsList.children).forEach(c => {
    if (c !== skillsEmpty) c.remove();
  });

  if (!skills.length) {
    skillsEmpty.style.display = '';
    return;
  }
  skillsEmpty.style.display = 'none';

  skills.forEach(skill => {
    const tile = document.createElement('div');
    tile.className = 'skill-tile';
    tile.innerHTML = `
      <div class="skill-tile-name">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/>
          <path d="M2 17l10 5 10-5"/>
          <path d="M2 12l10 5 10-5"/>
        </svg>
        ${escHtml(skill.name)}
      </div>
      ${skill.description ? `<div class="skill-tile-desc">${escHtml(skill.description)}</div>` : ''}
    `;
    skillsList.appendChild(tile);
  });
}

// ── Skill upload ───────────────────────────────────────────────────────────
const uploadInput  = document.getElementById('skill-upload');
const uploadStatus = document.getElementById('upload-status');

uploadInput.addEventListener('change', async () => {
  const file = uploadInput.files[0];
  if (!file) return;

  uploadStatus.textContent = 'Uploading…';
  uploadStatus.className = 'upload-status';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const r = await fetch('/api/skills/upload', {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Upload failed');
    uploadStatus.textContent = data.message || 'Uploaded';
    uploadStatus.className = 'upload-status ok';
    await loadSkills();
  } catch (err) {
    uploadStatus.textContent = err.message;
    uploadStatus.className = 'upload-status err';
  } finally {
    uploadInput.value = '';
    setTimeout(() => { uploadStatus.textContent = ''; uploadStatus.className = 'upload-status'; }, 5000);
  }
});

// ── Models ─────────────────────────────────────────────────────────────────
const modelsList  = document.getElementById('models-list');
const modelsEmpty = document.getElementById('models-empty');

let _activeModelId = null;

const PROVIDER_LABELS = {
  google_genai:  'Gemini',
  anthropic:     'Claude',
  openai:        'OpenAI',
  azure_openai:  'Azure',
  groq:          'Groq',
  mistralai:     'Mistral',
  ollama:        'Ollama',
};

async function loadModels() {
  const data = await api.get('/api/models');
  if (!data) return;
  _activeModelId = data.active_id;
  renderModels(data.models || [], data.active_id);
  updateSendBtn();
}

function renderModels(models, activeId) {
  Array.from(modelsList.children).forEach(c => {
    if (c !== modelsEmpty) c.remove();
  });

  if (!models.length) {
    modelsEmpty.style.display = '';
    return;
  }
  modelsEmpty.style.display = 'none';

  models.forEach(m => {
    const isActive = m.id === activeId;
    const label = PROVIDER_LABELS[m.provider] || m.provider;
    const card = document.createElement('div');
    card.className = `model-card${isActive ? ' active-model' : ''}`;
    card.dataset.id = m.id;
    card.innerHTML = `
      <div class="model-dot"></div>
      <div class="model-info">
        <div class="model-info-name">${escHtml(m.name)}</div>
        <div class="model-info-id">${escHtml(m.model)}</div>
      </div>
      <span class="provider-badge badge-${m.provider}">${escHtml(label)}</span>
      <button class="model-del" title="Remove" data-id="${m.id}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"/>
          <path d="M19 6l-1 14H6L5 6"/>
          <path d="M10 11v6"/><path d="M14 11v6"/>
          <path d="M9 6V4h6v2"/>
        </svg>
      </button>
    `;

    // Select on card click (not delete btn)
    card.addEventListener('click', async (e) => {
      if (e.target.closest('.model-del')) return;
      await api.put(`/api/models/${m.id}/select`);
      await loadModels();
    });

    card.querySelector('.model-del').addEventListener('click', async (e) => {
      e.stopPropagation();
      await api.del(`/api/models/${m.id}`);
      await loadModels();
    });

    modelsList.appendChild(card);
  });
}

// ── Model registration modal ────────────────────────────────────────────────
const backdrop   = document.getElementById('modal-backdrop');
const modelForm  = document.getElementById('model-form');
const formError  = document.getElementById('model-form-error');

const PROVIDER_DEFAULTS = {
  google_genai:  { model: 'gemini-2.5-flash',       name: 'Gemini Flash' },
  anthropic:     { model: 'claude-sonnet-4-6',       name: 'Claude Sonnet' },
  openai:        { model: 'gpt-4o',                  name: 'GPT-4o' },
  azure_openai:  { model: 'gpt-4o',                  name: 'Azure GPT-4o' },
  groq:          { model: 'llama3-70b-8192',         name: 'Groq Llama 3' },
  mistralai:     { model: 'mistral-large-latest',    name: 'Mistral Large' },
  ollama:        { model: 'llama3.2',                name: 'Ollama Llama 3' },
};

function openModal() { backdrop.classList.add('open'); onProviderChange(); }
function closeModal() { backdrop.classList.remove('open'); formError.textContent = ''; }

document.getElementById('add-model-btn').addEventListener('click', openModal);
document.getElementById('modal-close').addEventListener('click', closeModal);
document.getElementById('modal-cancel').addEventListener('click', closeModal);
backdrop.addEventListener('click', e => { if (e.target === backdrop) closeModal(); });

function onProviderChange() {
  const provider = document.getElementById('m-provider').value;
  const defaults = PROVIDER_DEFAULTS[provider] || {};

  document.getElementById('m-model').value = defaults.model || '';
  document.getElementById('m-name').value  = defaults.name  || '';

  const isAzure  = provider === 'azure_openai';
  const isOllama = provider === 'ollama';
  const needsKey = !isOllama;

  document.getElementById('field-apikey').style.display  = needsKey ? '' : 'none';
  document.getElementById('azure-fields').style.display  = isAzure  ? '' : 'none';
  document.getElementById('ollama-fields').style.display = isOllama ? '' : 'none';
}

document.getElementById('m-provider').addEventListener('change', onProviderChange);

modelForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const provider = document.getElementById('m-provider').value;
  const name     = document.getElementById('m-name').value.trim();
  const model    = document.getElementById('m-model').value.trim();
  const apiKey   = document.getElementById('m-apikey').value.trim() || null;

  if (!name || !model) { formError.textContent = 'Name and Model ID are required.'; return; }

  const extra = {};
  if (provider === 'azure_openai') {
    extra.azure_endpoint = document.getElementById('m-azure-endpoint').value.trim();
    extra.api_version    = document.getElementById('m-azure-version').value.trim() || '2025-01-01-preview';
  }
  if (provider === 'ollama') {
    extra.base_url = document.getElementById('m-base-url').value.trim() || 'http://localhost:11434';
  }

  const btn = document.getElementById('model-submit');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Registering…';
  formError.textContent = '';

  try {
    const res = await api.post('/api/models', { provider, model, api_key: apiKey, name, extra });
    if (!res) return;
    if (res.detail) throw new Error(res.detail);
    await loadModels();
    closeModal();
    modelForm.reset();
    onProviderChange();
  } catch (err) {
    formError.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Register';
  }
});

// ── Chat ───────────────────────────────────────────────────────────────────
const messagesEl = document.getElementById('messages');
const textarea   = document.getElementById('chat-input');
const sendBtn    = document.getElementById('send-btn');

let _streaming = false;
let _currentAiBubble = null;  // <div class="msg-bubble ai-bubble"> element being built
let _currentAiText   = '';

function updateSendBtn() {
  sendBtn.disabled = _streaming || _activeModelId === null;
}

// Auto-resize textarea
textarea.addEventListener('input', () => {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  sendBtn.disabled = _streaming || !textarea.value.trim() || _activeModelId === null;
});

textarea.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function appendUserMessage(text) {
  // Remove welcome screen if present
  const welcome = messagesEl.querySelector('.welcome');
  if (welcome) welcome.remove();

  const group = document.createElement('div');
  group.className = 'msg-group';
  group.innerHTML = `
    <div class="msg-row user">
      <div class="msg-avatar user-avatar">${escHtml(email[0]?.toUpperCase() || 'U')}</div>
      <div class="msg-bubble user-bubble">${escHtml(text)}</div>
    </div>
  `;
  messagesEl.appendChild(group);
  scrollToBottom();
}

function startAiMessage() {
  const group = document.createElement('div');
  group.className = 'msg-group';

  const row = document.createElement('div');
  row.className = 'msg-row';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar ai-avatar';
  avatar.textContent = 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble ai-bubble';
  bubble.innerHTML = `
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;

  row.appendChild(avatar);
  row.appendChild(bubble);
  group.appendChild(row);
  messagesEl.appendChild(group);
  _currentAiBubble = bubble;
  _currentAiText   = '';
  scrollToBottom();
}

function appendToken(text) {
  if (!_currentAiBubble) return;
  // Replace typing indicator on first token
  if (_currentAiText === '') {
    _currentAiBubble.innerHTML = '';
  }
  _currentAiText += text;
  _currentAiBubble.textContent = _currentAiText;
  scrollToBottom();
}

function appendToolCall(name) {
  const div = document.createElement('div');
  div.className = 'tool-event';
  div.innerHTML = `
    <div class="tool-call-line">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
      </svg>
      ${escHtml(name)}(…)
    </div>
  `;
  messagesEl.appendChild(div);
  scrollToBottom();
}

function appendToolResult(content) {
  const div = document.createElement('div');
  div.className = 'tool-event';
  div.innerHTML = `
    <div class="tool-result-line">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
      ${escHtml(content)}
    </div>
  `;
  messagesEl.appendChild(div);
  scrollToBottom();
}

function appendError(msg) {
  const div = document.createElement('div');
  div.className = 'tool-event';
  div.innerHTML = `
    <div class="error-line">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:14px;height:14px;flex-shrink:0">
        <circle cx="12" cy="12" r="10"/>
        <line x1="15" y1="9" x2="9" y2="15"/>
        <line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
      ${escHtml(msg)}
    </div>
  `;
  messagesEl.appendChild(div);
  scrollToBottom();
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage() {
  const text = textarea.value.trim();
  if (!text || _streaming) return;

  _streaming = true;
  updateSendBtn();
  textarea.value = '';
  textarea.style.height = 'auto';

  appendUserMessage(text);
  startAiMessage();

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ message: text }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${resp.status}`);
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();  // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') break;

        let event;
        try { event = JSON.parse(raw); } catch { continue; }

        switch (event.type) {
          case 'token':       appendToken(event.content);       break;
          case 'tool_call':   appendToolCall(event.name);       break;
          case 'tool_result': appendToolResult(event.content);  break;
          case 'error':       appendError(event.content);       break;
        }
      }
    }

    // If nothing was ever streamed, show a placeholder
    if (_currentAiText === '' && _currentAiBubble) {
      _currentAiBubble.innerHTML = '<em style="color:var(--muted)">No response</em>';
    }

  } catch (err) {
    if (_currentAiBubble) {
      _currentAiBubble.innerHTML = '';
    }
    appendError(err.message);
  } finally {
    _currentAiBubble = null;
    _currentAiText   = '';
    _streaming = false;
    updateSendBtn();
    textarea.focus();
  }
}

// ── Init ───────────────────────────────────────────────────────────────────
(async function init() {
  await Promise.all([loadSkills(), loadModels()]);
  textarea.focus();
})();
