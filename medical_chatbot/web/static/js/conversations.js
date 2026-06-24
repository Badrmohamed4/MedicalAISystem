let allConversations = [];
let activeConvId = null;
let sidebarOpen = false;

function toggleHistorySidebar() {
    sidebarOpen = !sidebarOpen;
    const sidebar = document.getElementById('history-sidebar');
    const overlay = document.getElementById('history-overlay');
    sidebar.classList.toggle('open', sidebarOpen);
    overlay.classList.toggle('visible', sidebarOpen);
    if (sidebarOpen) loadConversations();
}

async function loadConversations() {
    try {
        const res = await fetch('/api/conversations');
        const data = await res.json();
        allConversations = data.conversations || [];
        renderConversations(allConversations);
    } catch (e) {
        document.getElementById('history-list').innerHTML =
            '<div class="history-empty">Could not load conversations.</div>';
    }
}

function renderConversations(convs) {
    const list = document.getElementById('history-list');
    if (!convs.length) {
        list.innerHTML = '<div class="history-empty">No conversations yet.<br>Start chatting to create one.</div>';
        return;
    }
    list.innerHTML = convs.map(c => `
        <div class="conv-item ${c.session_id === activeConvId ? 'active' : ''}"
             id="conv-${c.session_id}"
             onclick="openConversation('${c.session_id}')">
            <div class="conv-item-body">
                <div class="conv-title" id="title-${c.session_id}">${escapeHtml(c.title || 'Untitled')}</div>
                <div class="conv-meta">${c.updated_at || ''}</div>
            </div>
            <div class="conv-actions">
                <button class="conv-action-btn" title="Rename"
                        onclick="startRename(event, '${c.session_id}')">&#9998;</button>
                <button class="conv-action-btn delete" title="Delete"
                        onclick="confirmDelete(event, '${c.session_id}')">&#128465;</button>
            </div>
        </div>
    `).join('');
}

function filterConversations(query) {
    const q = query.toLowerCase();
    const filtered = q
        ? allConversations.filter(c => (c.title || '').toLowerCase().includes(q))
        : allConversations;
    renderConversations(filtered);
}

async function openConversation(convId) {
    if (convId === activeConvId) {
        toggleHistorySidebar();
        return;
    }
    try {
        const res = await fetch('/api/conversations/' + convId);
        if (!res.ok) { alert('Could not load this conversation.'); return; }
        const data = await res.json();
        const chatHistory = document.getElementById('chat-history');
        chatHistory.innerHTML = '';
        data.history.forEach(msg => {
            appendMessage(msg.role, msg.content, false, msg.mode || 'patient');
        });
        activeConvId = convId;
        renderConversations(allConversations);
        toggleHistorySidebar();
        chatHistory.scrollTop = chatHistory.scrollHeight;
    } catch (e) {
        alert('Error loading conversation.');
    }
}

async function startNewConversation() {
    try {
        await fetch('/api/conversations/new', { method: 'POST' });
        activeConvId = null;
        const chatHistory = document.getElementById('chat-history');
        chatHistory.innerHTML = '';
        if (typeof loadHistory === 'function') loadHistory();
        toggleHistorySidebar();
    } catch (e) {
        alert('Could not start a new conversation.');
    }
}

function startRename(e, convId) {
    e.stopPropagation();
    const titleEl = document.getElementById('title-' + convId);
    const current = titleEl.innerText;
    const input = document.createElement('input');
    input.type = 'text';
    input.value = current;
    input.className = 'conv-rename-input';
    titleEl.replaceWith(input);
    input.focus();
    input.select();

    const finish = async () => {
        const newTitle = input.value.trim() || current;
        const div = document.createElement('div');
        div.id = 'title-' + convId;
        div.className = 'conv-title';
        div.innerText = newTitle;
        input.replaceWith(div);
        if (newTitle !== current) {
            await fetch('/api/conversations/' + convId + '/rename', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle })
            });
            const conv = allConversations.find(c => c.session_id === convId);
            if (conv) conv.title = newTitle;
        }
    };

    input.addEventListener('blur', finish);
    input.addEventListener('keydown', e => { if (e.key === 'Enter') input.blur(); });
}

async function confirmDelete(e, convId) {
    e.stopPropagation();
    if (!confirm('Delete this conversation? This cannot be undone.')) return;
    try {
        await fetch('/api/conversations/' + convId + '/delete', { method: 'POST' });
        allConversations = allConversations.filter(c => c.session_id !== convId);
        renderConversations(allConversations);
        if (convId === activeConvId) {
            activeConvId = null;
            document.getElementById('chat-history').innerHTML = '';
            if (typeof loadHistory === 'function') loadHistory();
        }
    } catch (e) {
        alert('Could not delete conversation.');
    }
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}