/**
 * Flyte WebSocket Client
 * Handles WebSocket connections and UI interactions
 */

let ws = null;
let reconnectAttempts = 0;
let maxReconnectAttempts = 5;
let reconnectDelay = 1000;

const elements = {
    messages: document.getElementById('messages'),
    status: document.getElementById('status'),
    statusText: document.querySelector('.status-text'),
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    connectButton: document.getElementById('connectButton'),
    connectionCount: document.getElementById('connectionCount')
};

/**
 * Add a message to the chat display
 */
function addMessage(text, type = 'received') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const timestamp = new Date().toLocaleTimeString();
    const timestampDiv = document.createElement('div');
    timestampDiv.className = 'timestamp';
    timestampDiv.textContent = timestamp;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';

    // Try to parse as JSON for better formatting
    try {
        const data = JSON.parse(text);
        contentDiv.textContent = data.message || JSON.stringify(data, null, 2);

        // Update connection count if available
        if (data.connections !== undefined) {
            elements.connectionCount.textContent = data.connections;
        }
    } catch (e) {
        contentDiv.textContent = text;
    }

    messageDiv.appendChild(timestampDiv);
    messageDiv.appendChild(contentDiv);
    elements.messages.appendChild(messageDiv);

    // Auto-scroll to bottom
    elements.messages.scrollTop = elements.messages.scrollHeight;
}

/**
 * Update connection status UI
 */
function updateStatus(connected) {
    if (connected) {
        elements.statusText.textContent = 'Connected';
        elements.status.className = 'status connected';
        elements.messageInput.disabled = false;
        elements.sendButton.disabled = false;
        elements.connectButton.textContent = 'Disconnect';
        elements.connectButton.classList.add('disconnect');
        reconnectAttempts = 0;
    } else {
        elements.statusText.textContent = 'Disconnected';
        elements.status.className = 'status disconnected';
        elements.messageInput.disabled = true;
        elements.sendButton.disabled = true;
        elements.connectButton.textContent = 'Connect';
        elements.connectButton.classList.remove('disconnect');
    }
}

/**
 * Connect to WebSocket server
 */
function connect() {
    // Get WebSocket URL (injected by server or use default)
    const wsUrl = window.WS_URL || `ws://${window.location.host}/ws`;

    addMessage(`Connecting to ${wsUrl}...`, 'system');

    ws = new WebSocket(wsUrl);

    ws.onopen = function(event) {
        addMessage('✅ Connected to server', 'system');
        updateStatus(true);
    };

    ws.onmessage = function(event) {
        addMessage(event.data, 'received');
    };

    ws.onerror = function(error) {
        addMessage('❌ WebSocket error occurred', 'system');
        console.error('WebSocket error:', error);
    };

    ws.onclose = function(event) {
        addMessage('🔌 Disconnected from server', 'system');
        updateStatus(false);
        ws = null;

        // Auto-reconnect logic
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = reconnectDelay * reconnectAttempts;
            addMessage(`🔄 Reconnecting in ${delay / 1000} seconds... (Attempt ${reconnectAttempts}/${maxReconnectAttempts})`, 'system');
            setTimeout(() => {
                if (!ws || ws.readyState === WebSocket.CLOSED) {
                    connect();
                }
            }, delay);
        } else {
            addMessage('❌ Max reconnection attempts reached. Click Connect to try again.', 'system');
            reconnectAttempts = 0;
        }
    };
}

/**
 * Disconnect from WebSocket server
 */
function disconnect() {
    if (ws) {
        reconnectAttempts = maxReconnectAttempts; // Prevent auto-reconnect
        ws.close();
        addMessage('👋 Manually disconnected', 'system');
    }
}

/**
 * Toggle connection state
 */
function toggleConnection() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        disconnect();
    } else {
        reconnectAttempts = 0; // Reset counter for manual connect
        connect();
    }
}

/**
 * Send a message through WebSocket
 */
function sendMessage() {
    const message = elements.messageInput.value.trim();

    if (!message) {
        return;
    }

    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addMessage('❌ Not connected to server', 'system');
        return;
    }

    try {
        ws.send(message);
        addMessage(message, 'sent');
        elements.messageInput.value = '';
        elements.messageInput.focus();
    } catch (error) {
        addMessage('❌ Failed to send message', 'system');
        console.error('Send error:', error);
    }
}

/**
 * Clear all messages
 */
function clearMessages() {
    elements.messages.innerHTML = '';
    addMessage('🗑️ Messages cleared', 'system');
}

/**
 * Handle Enter key press in message input
 */
elements.messageInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
});

/**
 * Fetch connection count periodically
 */
function updateConnectionCount() {
    fetch('/connections')
        .then(response => response.json())
        .then(data => {
            elements.connectionCount.textContent = data.active_connections;
        })
        .catch(error => {
            console.error('Failed to fetch connection count:', error);
        });
}

/**
 * Initialize the application
 */
function init() {
    addMessage('👋 Welcome to Flyte WebSocket Demo!', 'system');
    addMessage('Click "Connect" to establish a WebSocket connection.', 'system');

    // Update connection count every 5 seconds
    setInterval(updateConnectionCount, 5000);
}

// Start the app
init();
