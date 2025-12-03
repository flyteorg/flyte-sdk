/**
 * Flyte Speech Transcription Client
 * Handles audio recording, WebSocket streaming, and UI updates
 */

let ws = null;
let audioContext = null;
let audioProcessor = null;
let mediaStream = null;
let isRecording = false;
let wordCount = 0;

const elements = {
    status: document.getElementById('status'),
    statusText: document.querySelector('.status-text'),
    connectBtn: document.getElementById('connectBtn'),
    recordBtn: document.getElementById('recordBtn'),
    transcription: document.getElementById('transcription'),
    wordCount: document.getElementById('wordCount'),
    audioLevelFill: document.getElementById('audioLevelFill')
};

/**
 * Update connection status UI
 */
function updateStatus(state, text) {
    elements.status.className = `status ${state}`;
    elements.statusText.textContent = text;
}

/**
 * Toggle WebSocket connection
 */
function toggleConnection() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        disconnect();
    } else {
        connect();
    }
}

/**
 * Connect to WebSocket server
 */
function connect() {
    const wsUrl = window.WS_URL || `ws://${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = function() {
        updateStatus('connected', 'Connected - Ready to record');
        elements.connectBtn.innerHTML = '<span class="btn-icon">🔌</span> Disconnect';
        elements.connectBtn.className = 'btn btn-danger';
        elements.recordBtn.disabled = false;
    };

    ws.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'transcription' && data.text) {
                addTranscription(data.text, data.timestamp);
            } else if (data.type === 'system') {
                console.log('System message:', data.message);
            } else if (data.type === 'error') {
                console.error('Transcription error:', data.message);
                showError(data.message);
            }
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateStatus('disconnected', 'Connection error');
        showError('WebSocket connection failed');
    };

    ws.onclose = function() {
        updateStatus('disconnected', 'Disconnected');
        elements.connectBtn.innerHTML = '<span class="btn-icon">🔌</span> Connect';
        elements.connectBtn.className = 'btn btn-primary';
        elements.recordBtn.disabled = true;

        if (isRecording) {
            stopRecording();
        }
    };
}

/**
 * Disconnect from WebSocket
 */
function disconnect() {
    if (isRecording) {
        stopRecording();
    }
    if (ws) {
        ws.close();
        ws = null;
    }
}

/**
 * Toggle recording
 */
async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

/**
 * Start audio recording
 */
async function startRecording() {
    try {
        // Request microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });

        // Create audio context
        audioContext = new AudioContext({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(mediaStream);

        // Create script processor for audio processing
        audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);

        audioProcessor.onaudioprocess = function(e) {
            if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) {
                return;
            }

            const inputData = e.inputBuffer.getChannelData(0);

            // Calculate audio level for visualization
            const sum = inputData.reduce((acc, val) => acc + Math.abs(val), 0);
            const avgLevel = (sum / inputData.length) * 100;
            updateAudioLevel(avgLevel);

            // Convert float32 (-1.0 to 1.0) to int16 (-32768 to 32767)
            const int16Data = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const sample = Math.max(-1, Math.min(1, inputData[i]));
                int16Data[i] = sample < 0 ? sample * 32768 : sample * 32767;
            }

            // Send to WebSocket
            ws.send(int16Data.buffer);
        };

        // Connect nodes
        source.connect(audioProcessor);
        audioProcessor.connect(audioContext.destination);

        isRecording = true;
        updateStatus('recording', 'Recording...');
        elements.recordBtn.innerHTML = '<span class="btn-icon">⏸️</span> Stop Recording';
        elements.recordBtn.className = 'btn btn-danger';

    } catch (error) {
        console.error('Error starting recording:', error);
        showError('Could not access microphone. Please grant permission.');
    }
}

/**
 * Stop audio recording
 */
function stopRecording() {
    isRecording = false;

    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
    }

    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    updateStatus('connected', 'Connected - Ready to record');
    elements.recordBtn.innerHTML = '<span class="btn-icon">🎙️</span> Start Recording';
    elements.recordBtn.className = 'btn btn-primary';
    updateAudioLevel(0);
}

/**
 * Update audio level visualization
 */
function updateAudioLevel(level) {
    elements.audioLevelFill.style.width = `${Math.min(100, level * 2)}%`;
}

/**
 * Add transcription to display
 */
function addTranscription(text, timestamp) {
    if (!text.trim()) return;

    // Remove placeholder if present
    const placeholder = elements.transcription.querySelector('.placeholder');
    if (placeholder) {
        elements.transcription.innerHTML = '';
    }

    // Create transcript line
    const lineDiv = document.createElement('div');
    lineDiv.className = 'transcript-line';

    // Time
    const timeDiv = document.createElement('div');
    timeDiv.className = 'time';
    timeDiv.textContent = new Date(timestamp || Date.now()).toLocaleTimeString();

    // Extract speaker if present
    const speakerMatch = text.match(/^\[Speaker (\d+)\]\s*/);
    let displayText = text;
    let speaker = null;

    if (speakerMatch) {
        speaker = speakerMatch[1];
        displayText = text.replace(speakerMatch[0], '');
    }

    // Text
    const textDiv = document.createElement('div');
    textDiv.className = 'text';

    if (speaker) {
        const speakerSpan = document.createElement('span');
        speakerSpan.className = 'speaker';
        speakerSpan.textContent = `Speaker ${speaker}`;
        textDiv.appendChild(speakerSpan);
    }

    const textSpan = document.createElement('span');
    textSpan.textContent = displayText;
    textDiv.appendChild(textSpan);

    // Assemble
    lineDiv.appendChild(timeDiv);
    lineDiv.appendChild(textDiv);

    elements.transcription.appendChild(lineDiv);

    // Auto-scroll
    elements.transcription.scrollTop = elements.transcription.scrollHeight;

    // Update word count
    updateWordCount(displayText);
}

/**
 * Update word count
 */
function updateWordCount(text) {
    const words = text.trim().split(/\s+/).filter(w => w.length > 0);
    wordCount += words.length;
    elements.wordCount.textContent = `${wordCount} word${wordCount !== 1 ? 's' : ''}`;
}

/**
 * Clear transcription
 */
function clearTranscription() {
    elements.transcription.innerHTML = '<p class="placeholder">Transcription will appear here...</p>';
    wordCount = 0;
    elements.wordCount.textContent = '0 words';
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'transcript-line';
    errorDiv.style.borderLeftColor = '#dc3545';
    errorDiv.style.background = '#f8d7da';

    const textDiv = document.createElement('div');
    textDiv.className = 'text';
    textDiv.style.color = '#721c24';
    textDiv.textContent = `⚠️ Error: ${message}`;

    errorDiv.appendChild(textDiv);
    elements.transcription.appendChild(errorDiv);
    elements.transcription.scrollTop = elements.transcription.scrollHeight;
}

/**
 * Initialize the application
 */
function init() {
    console.log('Flyte Speech Transcription initialized');

    // Check for WebSocket support
    if (!window.WebSocket) {
        showError('WebSocket is not supported by your browser');
        return;
    }

    // Check for getUserMedia support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('Your browser does not support audio recording');
        return;
    }
}

// Start the app
init();
