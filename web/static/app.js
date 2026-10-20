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
};

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

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

// ── File Explorer ──────────────────────────────────────────────────────────
const feTreeEl   = document.getElementById('fe-tree');
const feEmptyEl  = document.getElementById('fe-empty');
const uploadStatus = document.getElementById('upload-status');

const FILE_ICONS = {
  '.md':   '📝', '.txt':  '📄', '.py':   '🐍',
  '.js':   '⚡', '.ts':   '⚡', '.json': '{}',
  '.html': '🌐', '.css':  '🎨', '.scss': '🎨',
  '.zip':  '📦', '.pdf':  '📕',
  '.png':  '🖼', '.jpg':  '🖼', '.jpeg': '🖼', '.gif':  '🖼', '.svg': '🖼',
  '.csv':  '📊', '.xlsx': '📊', '.xls':  '📊',
  '.sh':   '⚙', '.yaml': '⚙', '.yml':  '⚙', '.toml': '⚙',
  '.env':  '🔑', '.gitignore': '🚫',
};

function getFileIcon(ext, name) {
  if (name === 'SKILL.md' || name === 'skill.md') return '⚡';
  return FILE_ICONS[ext.toLowerCase()] || '📄';
}

async function loadFileTree() {
  const data = await api.get('/api/files');
  if (!data) return;
  renderFileTree(data.tree || []);
}

function renderFileTree(tree) {
  feTreeEl.innerHTML = '';
  if (!tree.length) {
    feEmptyEl.style.display = '';
    return;
  }
  feEmptyEl.style.display = 'none';
  renderNodes(tree, feTreeEl, 0, true);
}

function renderNodes(nodes, parent, depth, expandTopLevel) {
  nodes.forEach(node => {
    if (node.type === 'dir') {
      renderDirNode(node, parent, depth, expandTopLevel);
    } else {
      renderFileNode(node, parent, depth);
    }
  });
}

function renderDirNode(node, parent, depth, autoExpand) {
  const row = document.createElement('div');
  row.className = 'fe-node fe-dir';
  row.style.setProperty('--depth', depth);

  row.innerHTML = `
    <span class="fe-toggle">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polyline points="9 18 15 12 9 6"/>
      </svg>
    </span>
    <span class="fe-icon">📁</span>
    <span class="fe-name">${escHtml(node.name)}</span>
  `;

  const childContainer = document.createElement('div');
  childContainer.className = 'fe-children';

  if (node.children && node.children.length) {
    renderNodes(node.children, childContainer, depth + 1, false);
  }

  const isOpen = autoExpand;
  if (isOpen) {
    row.classList.add('open');
  } else {
    childContainer.hidden = true;
  }

  row.addEventListener('click', () => {
    const nowOpen = row.classList.toggle('open');
    childContainer.hidden = !nowOpen;
  });

  parent.appendChild(row);
  parent.appendChild(childContainer);
}

function renderFileNode(node, parent, depth) {
  const row = document.createElement('div');
  row.className = 'fe-node fe-file';
  row.style.setProperty('--depth', depth);
  row.dataset.path = node.path;

  const icon = getFileIcon(node.ext || '', node.name);
  const size = node.size < 1024 ? `${node.size} B`
    : node.size < 1048576 ? `${(node.size/1024).toFixed(1)} KB`
    : `${(node.size/1048576).toFixed(1)} MB`;

  row.innerHTML = `
    <span class="fe-toggle"></span>
    <span class="fe-icon">${icon}</span>
    <span class="fe-name">${escHtml(node.name)}</span>
    ${node.is_binary ? `<span class="fe-badge">${escHtml(size)}</span>` : ''}
  `;

  row.title = node.is_binary ? `${node.name} · binary (${size})` : `${node.name} · click to edit`;

  if (!node.is_binary) {
    row.addEventListener('click', () => {
      document.querySelectorAll('.fe-node.selected').forEach(n => n.classList.remove('selected'));
      row.classList.add('selected');
      openFile(node.path, node.name);
    });
  } else {
    row.style.opacity = '0.55';
    row.style.cursor = 'default';
  }

  parent.appendChild(row);
}

// ── File Editor ────────────────────────────────────────────────────────────
const editorBackdrop   = document.getElementById('editor-backdrop');
const editorTextarea   = document.getElementById('editor-textarea');
const editorBreadcrumb = document.getElementById('editor-breadcrumb');
const editorStatusMsg  = document.getElementById('editor-status-msg');
const versionsDropdown = document.getElementById('versions-dropdown');
const versionsBtn      = document.getElementById('versions-btn');
const versionsMenu     = document.getElementById('versions-menu');

let _editorPath     = null;
let _editorModified = false;

async function openFile(path, name) {
  editorStatusMsg.textContent = 'Loading…';
  editorStatusMsg.className = 'editor-status-msg dim';
  editorBackdrop.classList.add('open');

  const data = await api.get(`/api/files/content?path=${encodeURIComponent(path)}`);
  if (!data) { editorBackdrop.classList.remove('open'); return; }

  _editorPath     = path;
  _editorModified = false;

  // Breadcrumb
  const parts = path.split('/');
  editorBreadcrumb.innerHTML = parts.map((p, i) =>
    i === parts.length - 1
      ? `<span class="crumb active">${escHtml(p)}</span>`
      : `<span class="crumb">${escHtml(p)}</span><span class="crumb-sep">/</span>`
  ).join('');

  editorTextarea.value = data.content || '';
  editorStatusMsg.textContent = '';

  renderVersions(data.versions || [], path);
  editorTextarea.focus();
}

function renderVersions(versions, path) {
  // Remove old version items (keep the header)
  const header = versionsMenu.querySelector('.versions-menu-header');
  versionsMenu.innerHTML = '';
  if (header) versionsMenu.appendChild(header);

  if (!versions.length) {
    versionsBtn.textContent = 'No history';
    versionsBtn.disabled = true;
    const empty = document.createElement('div');
    empty.style.cssText = 'padding:14px;text-align:center;font-size:0.8125rem;color:var(--muted)';
    empty.textContent = 'No saved versions yet';
    versionsMenu.appendChild(empty);
    return;
  }

  versionsBtn.textContent = `History (${versions.length}) ▾`;
  versionsBtn.disabled = false;

  // Newest first
  [...versions].reverse().forEach(v => {
    const kb = (v.size / 1024).toFixed(1);
    const item = document.createElement('div');
    item.className = 'versions-item';
    item.innerHTML = `
      <div class="version-info">
        <div class="version-label">${escHtml(v.label)}</div>
        <div class="version-meta">${escHtml(v.filename)} · ${kb} KB</div>
      </div>
      <button class="btn-ghost btn-sm" data-filename="${escHtml(v.filename)}">Restore</button>
    `;
    item.querySelector('button').addEventListener('click', async (e) => {
      e.stopPropagation();
      versionsMenu.hidden = true;
      if (!confirm(`Restore ${v.label}? The current version will be backed up first.`)) return;
      await restoreVersion(path, v.filename);
    });
    versionsMenu.appendChild(item);
  });
}

versionsBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  versionsMenu.hidden = !versionsMenu.hidden;
});
document.addEventListener('click', () => { versionsMenu.hidden = true; });

async function saveFile() {
  if (!_editorPath) return;
  editorStatusMsg.textContent = 'Saving…';
  editorStatusMsg.className = 'editor-status-msg dim';

  try {
    const res = await api.post('/api/files/save', { path: _editorPath, content: editorTextarea.value });
    if (!res || res.detail) throw new Error(res?.detail || 'Save failed');

    editorStatusMsg.textContent = '✓ Saved';
    editorStatusMsg.className = 'editor-status-msg ok';
    _editorModified = false;

    // Refresh breadcrumb (remove unsaved indicator)
    const parts = _editorPath.split('/');
    editorBreadcrumb.innerHTML = parts.map((p, i) =>
      i === parts.length - 1
        ? `<span class="crumb active">${escHtml(p)}</span>`
        : `<span class="crumb">${escHtml(p)}</span><span class="crumb-sep">/</span>`
    ).join('');

    renderVersions(res.versions || [], _editorPath);
    await loadFileTree();
    setTimeout(() => { if (editorStatusMsg.textContent === '✓ Saved') editorStatusMsg.textContent = ''; }, 3000);
  } catch (err) {
    editorStatusMsg.textContent = `Error: ${err.message}`;
    editorStatusMsg.className = 'editor-status-msg err';
  }
}

async function restoreVersion(path, versionFilename) {
  try {
    const res = await api.post('/api/files/restore', { path, version_name: versionFilename });
    if (!res?.ok) throw new Error('Restore failed');
    await openFile(path, path.split('/').pop());
    editorStatusMsg.textContent = '✓ Restored';
    editorStatusMsg.className = 'editor-status-msg ok';
    setTimeout(() => { if (editorStatusMsg.textContent === '✓ Restored') editorStatusMsg.textContent = ''; }, 3000);
  } catch (err) {
    editorStatusMsg.textContent = `Error: ${err.message}`;
    editorStatusMsg.className = 'editor-status-msg err';
  }
}

document.getElementById('editor-save-btn').addEventListener('click', saveFile);

document.getElementById('editor-close').addEventListener('click', () => {
  if (_editorModified && !confirm('You have unsaved changes. Close anyway?')) return;
  editorBackdrop.classList.remove('open');
  _editorPath = null;
  document.querySelectorAll('.fe-node.selected').forEach(n => n.classList.remove('selected'));
});

document.getElementById('editor-delete-btn').addEventListener('click', async () => {
  if (!_editorPath) return;
  const name = _editorPath.split('/').pop();
  if (!confirm(`Delete "${name}"? This cannot be undone.`)) return;

  const r = await fetch(`/api/files?path=${encodeURIComponent(_editorPath)}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` },
  });
  const data = await r.json().catch(() => ({}));
  if (data.ok) {
    editorBackdrop.classList.remove('open');
    _editorPath = null;
    await loadFileTree();
  } else {
    editorStatusMsg.textContent = `Error: ${data.detail || 'Delete failed'}`;
    editorStatusMsg.className = 'editor-status-msg err';
  }
});

editorTextarea.addEventListener('input', () => {
  if (!_editorModified) {
    _editorModified = true;
    const active = editorBreadcrumb.querySelector('.crumb.active');
    if (active && !active.textContent.endsWith(' ●')) {
      active.textContent += ' ●';
    }
  }
});

editorTextarea.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    saveFile();
  }
  if (e.key === 'Tab') {
    e.preventDefault();
    const start = e.target.selectionStart;
    const end   = e.target.selectionEnd;
    e.target.value = e.target.value.substring(0, start) + '  ' + e.target.value.substring(end);
    e.target.selectionStart = e.target.selectionEnd = start + 2;
  }
});

// ── File Upload ────────────────────────────────────────────────────────────
const fileUploadInput   = document.getElementById('file-upload');
const uploadBackdrop    = document.getElementById('upload-backdrop');
const uploadFilenameEl  = document.getElementById('upload-filename');

let _pendingUploadFile = null;

fileUploadInput.addEventListener('change', () => {
  const file = fileUploadInput.files[0];
  if (!file) return;
  _pendingUploadFile = file;
  uploadFilenameEl.textContent = file.name;
  document.getElementById('upload-dest').value = '';
  uploadBackdrop.classList.add('open');
});

function closeUploadModal() {
  uploadBackdrop.classList.remove('open');
  fileUploadInput.value = '';
  _pendingUploadFile = null;
}

document.getElementById('upload-modal-close').addEventListener('click', closeUploadModal);
document.getElementById('upload-modal-cancel').addEventListener('click', closeUploadModal);
uploadBackdrop.addEventListener('click', e => { if (e.target === uploadBackdrop) closeUploadModal(); });

document.getElementById('upload-modal-confirm').addEventListener('click', async () => {
  if (!_pendingUploadFile) return;
  const fileToUpload = _pendingUploadFile;   // capture before closeUploadModal nulls it
  const dest = document.getElementById('upload-dest').value.trim();
  closeUploadModal();

  uploadStatus.textContent = 'Uploading…';
  uploadStatus.className = 'upload-status';

  const formData = new FormData();
  formData.append('file', fileToUpload);
  if (dest) formData.append('dest_folder', dest);

  try {
    const r = await fetch('/api/upload', {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = typeof data.detail === 'string' ? data.detail
        : Array.isArray(data.detail) ? data.detail.map(e => e.msg).join(', ')
        : `Upload failed (${r.status})`;
      throw new Error(msg);
    }
    uploadStatus.textContent = data.message || 'Uploaded';
    uploadStatus.className = 'upload-status ok';
    await loadFileTree();
  } catch (err) {
    uploadStatus.textContent = err.message;
    uploadStatus.className = 'upload-status err';
  } finally {
    setTimeout(() => { uploadStatus.textContent = ''; uploadStatus.className = 'upload-status'; }, 5000);
  }
});

// ── New Folder ─────────────────────────────────────────────────────────────
const newfolderBackdrop = document.getElementById('newfolder-backdrop');

document.getElementById('new-folder-btn').addEventListener('click', () => {
  document.getElementById('newfolder-path').value = '';
  newfolderBackdrop.classList.add('open');
  setTimeout(() => document.getElementById('newfolder-path').focus(), 50);
});

function closeNewfolderModal() { newfolderBackdrop.classList.remove('open'); }
document.getElementById('newfolder-close').addEventListener('click', closeNewfolderModal);
document.getElementById('newfolder-cancel').addEventListener('click', closeNewfolderModal);
newfolderBackdrop.addEventListener('click', e => { if (e.target === newfolderBackdrop) closeNewfolderModal(); });

document.getElementById('newfolder-confirm').addEventListener('click', async () => {
  const path = document.getElementById('newfolder-path').value.trim();
  if (!path) return;
  closeNewfolderModal();

  uploadStatus.textContent = 'Creating folder…';
  uploadStatus.className = 'upload-status';
  try {
    const res = await api.post('/api/files/mkdir', { path });
    if (!res?.ok) throw new Error(res?.detail || 'Failed');
    uploadStatus.textContent = `Folder created`;
    uploadStatus.className = 'upload-status ok';
    await loadFileTree();
  } catch (err) {
    uploadStatus.textContent = err.message;
    uploadStatus.className = 'upload-status err';
  } finally {
    setTimeout(() => { uploadStatus.textContent = ''; uploadStatus.className = 'upload-status'; }, 4000);
  }
});

document.getElementById('newfolder-path').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') document.getElementById('newfolder-confirm').click();
});

// ── Folder Upload ──────────────────────────────────────────────────────────
const folderUploadInput = document.getElementById('folder-upload');

folderUploadInput.addEventListener('change', async () => {
  const files = Array.from(folderUploadInput.files);
  if (!files.length) return;
  folderUploadInput.value = '';

  uploadStatus.textContent = `Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`;
  uploadStatus.className = 'upload-status';

  let succeeded = 0, failed = 0;
  for (const file of files) {
    const relPath = file.webkitRelativePath || file.name;
    const parts = relPath.split('/');
    const destFolder = parts.slice(0, -1).join('/');
    const formData = new FormData();
    formData.append('file', file, parts[parts.length - 1]);
    if (destFolder) formData.append('dest_folder', destFolder);
    try {
      const r = await fetch('/api/upload', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (r.ok) succeeded++;
      else failed++;
    } catch { failed++; }
  }

  uploadStatus.textContent = failed > 0
    ? `Uploaded ${succeeded}/${files.length} files (${failed} failed)`
    : `Uploaded ${succeeded} file${succeeded > 1 ? 's' : ''}`;
  uploadStatus.className = failed > 0 ? 'upload-status err' : 'upload-status ok';
  await loadFileTree();
  setTimeout(() => { uploadStatus.textContent = ''; uploadStatus.className = 'upload-status'; }, 4000);
});

// ── Download Session ───────────────────────────────────────────────────────
document.getElementById('download-session-btn').addEventListener('click', () => {
  fetchDownload('/api/session/download');
});

async function fetchDownload(url) {
  uploadStatus.textContent = 'Preparing download…';
  uploadStatus.className = 'upload-status';
  try {
    const r = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    if (!r.ok) throw new Error('Download failed');
    const blob = await r.blob();
    const cdHeader = r.headers.get('Content-Disposition') || '';
    const nameMatch = cdHeader.match(/filename="?([^"]+)"?/);
    const filename = nameMatch ? nameMatch[1] : 'session.zip';
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = objUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(objUrl);
    uploadStatus.textContent = '✓ Downloaded';
    uploadStatus.className = 'upload-status ok';
  } catch (err) {
    uploadStatus.textContent = err.message;
    uploadStatus.className = 'upload-status err';
  } finally {
    setTimeout(() => { uploadStatus.textContent = ''; uploadStatus.className = 'upload-status'; }, 4000);
  }
}

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

    card.addEventListener('click', async (e) => {
      if (e.target.closest('.model-del')) return;
      await api.put(`/api/models/${m.id}/select`);
      await loadModels();
    });

    card.querySelector('.model-del').addEventListener('click', async (e) => {
      e.stopPropagation();
      const r = await fetch(`/api/models/${m.id}`, { method: 'DELETE', headers: api.headers() });
      await r.json();
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
let _currentAiBubble = null;
let _currentAiText   = '';

function updateSendBtn() {
  sendBtn.disabled = _streaming || _activeModelId === null;
}

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

function appendUserMessage(text) {
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

  // Append session context note to the actual API message (not shown to user)
  const messageWithContext = text + '\n\nPlease note: all required files are available in the session folder.';

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ message: messageWithContext }),
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
      buffer = lines.pop();

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
  await Promise.all([loadFileTree(), loadModels()]);
  textarea.focus();
})();
