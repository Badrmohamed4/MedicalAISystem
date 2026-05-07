const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');
const imageUpload = document.getElementById('image-upload');
let currentMode = 'patient'; // Track current mode

// Load chat history on page load
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        currentMode = data.mode || 'patient';
        // Set UI to correct mode without notifying backend (suppress "Switched" msg)
        switchMode(currentMode, false);

        // Clear existing messages (except initial greeting if no history)
        if (data.history && data.history.length > 0) {
            chatHistory.innerHTML = '';

            // Display each message from history
            data.history.forEach(msg => {
                // Use the mode from history, fallback to current if missing
                appendMessage(msg.role, msg.content, false, msg.mode || 'patient');
            });
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// Call loadHistory when page loads
document.addEventListener('DOMContentLoaded', loadHistory);

userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

imageUpload.addEventListener('change', function (e) {
    if (this.files && this.files[0]) {
        uploadImage(this.files[0]);
    }
});

function appendMessage(role, text, isImage = false, mode = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    // Use provided mode, or default to currentMode
    const msgMode = mode || currentMode;
    msgDiv.setAttribute('data-mode', msgMode);

    let contentHtml = '';
    if (isImage) {
        contentHtml = `<div class="bubble"><img src="${text}" class="chat-image"><br><em>Image Uploaded</em></div>`;
    } else {
        // Simple markdown parsing
        let formattedText = text.replace(/\n/g, '<br>');
        formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        contentHtml = `<div class="bubble">${formattedText}</div>`;
    }

    msgDiv.innerHTML = contentHtml;
    chatHistory.appendChild(msgDiv);

    // Only show if matches current mode (for initial load)
    if (msgMode !== currentMode) {
        msgDiv.style.display = 'none';
    } else {
        msgDiv.style.display = 'flex';
    }

    chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    userInput.value = '';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });
        const data = await response.json();
        appendMessage('system', data.response);
    } catch (error) {
        appendMessage('system', 'Error: Could not connect to server.');
    }
}

async function uploadImage(file) {
    appendMessage('user', URL.createObjectURL(file), true);

    // Show analyzing message
    const analyzingMsg = document.createElement('div');
    analyzingMsg.className = 'message system analyzing';
    analyzingMsg.setAttribute('data-mode', currentMode);
    analyzingMsg.innerHTML = '<div class="bubble">🔍 Analyzing image... Please wait.</div>';
    chatHistory.appendChild(analyzingMsg);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        // Remove analyzing message
        analyzingMsg.remove();

        appendMessage('system', data.response);
    } catch (error) {
        // Remove analyzing message on error
        analyzingMsg.remove();
        appendMessage('system', 'Error: Upload failed.');
    }
}

async function switchMode(mode) {
    currentMode = mode; // Update current mode

    // Helper to safely toggle active class
    const updateActive = (id) => {
        const el = document.getElementById(id);
        if (el) {
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            el.classList.add('active');
        }
    };
    updateActive(`btn-${mode}`);

    // Update headers
    const title = document.getElementById('chat-title');
    const badge = document.getElementById('mode-badge');

    if (title && badge) {
        if (mode === 'patient') {
            title.innerText = 'Patient Assistant';
            badge.innerText = 'Triage Mode';
        } else {
            title.innerText = 'Doctor Dashboard';
            badge.innerText = 'Clinical Mode';
        }
    }

    // Filter messages by mode
    document.querySelectorAll('.message').forEach(msg => {
        if (msg.getAttribute('data-mode') === mode) {
            msg.style.display = 'flex'; // Show messages for this mode
        } else {
            msg.style.display = 'none'; // Hide messages from other mode
        }
    });

    // Call backend API
    if (notifyBackend) {
        try {
            const response = await fetch('/api/switch_mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            });
            const data = await response.json();
            appendMessage('system', data.response);
        } catch (e) {
            console.error(e);
        }
    }
}

async function clearSession() {
    if (confirm("Are you sure you want to clear the session? History will be lost.")) {
        await fetch('/api/clear', { method: 'POST' });
        location.reload();
    }
}

async function setContext(context) {
    // Update UI
    document.querySelectorAll('.context-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-${context}`).classList.add('active');

    // Send to backend
    try {
        await fetch('/api/set_context', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ context: context })
        });
        console.log(`Context set to: ${context}`);
    } catch (e) {
        console.error('Failed to set context:', e);
    }
}
