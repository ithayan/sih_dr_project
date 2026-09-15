const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const runBtn = document.getElementById('run-btn');
const vizZone = document.getElementById('viz-zone');
const rawImage = document.getElementById('raw-image');
const xaiCanvas = document.getElementById('xai-canvas');
const consoleLog = document.getElementById('console');

const gradeResult = document.getElementById('grade-result');
const foveaResult = document.getElementById('fovea-result');
const latencyResult = document.getElementById('latency-result');

let currentBase64 = null;

// Helpers
function logToConsole(msg, isError = false) {
    const timestamp = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    if (isError) div.className = 'error';
    div.innerText = `[${timestamp}] ${msg}`;
    consoleLog.appendChild(div);
    consoleLog.scrollTop = consoleLog.scrollHeight;
}

// Upload Handling
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--accent)';
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--border)';
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--border)';
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        logToConsole('Error: Please upload a valid image file.', true);
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        // Base64 string looks like "data:image/jpeg;base64,/9j/4AAQSk..."
        const fullDataUrl = e.target.result;
        currentBase64 = fullDataUrl.split(',')[1]; // Extract just the base64 part
        
        rawImage.src = fullDataUrl;
        
        uploadZone.classList.add('hidden');
        vizZone.classList.remove('hidden');
        runBtn.disabled = false;
        
        // Clear previous results
        const ctx = xaiCanvas.getContext('2d');
        ctx.clearRect(0, 0, xaiCanvas.width, xaiCanvas.height);
        
        gradeResult.innerText = 'Pending';
        gradeResult.className = 'pending';
        foveaResult.innerText = 'Pending';
        latencyResult.innerText = '-- ms';
        
        logToConsole(`Ingested file: ${file.name}`);
        logToConsole(`Ready for Quantum Inference.`);
    };
    reader.readAsDataURL(file);
}

// Ensure canvas matches image size once loaded
rawImage.addEventListener('load', () => {
    xaiCanvas.width = rawImage.width;
    xaiCanvas.height = rawImage.height;
});

// Inference Handling
runBtn.addEventListener('click', async () => {
    if (!currentBase64) return;
    
    runBtn.disabled = true;
    logToConsole('Initiating API call to Quantum Microservice...');
    
    const startTime = performance.now();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_base64: currentBase64 })
        });
        
        const data = await response.json();
        const endTime = performance.now();
        const latency = (endTime - startTime).toFixed(2);
        
        if (!response.ok) throw new Error(data.error || 'Server error');
        
        logToConsole('Gate 1 [Gradability]: Passed');
        
        if (data.severity_grade === 0) {
            logToConsole('Gate 2 [Anomaly]: Normal. No Referable DR.');
            gradeResult.innerText = 'Grade 0 (Normal)';
            gradeResult.className = 'normal';
        } else {
            logToConsole(`Gate 2 [Anomaly]: Referable DR Detected (Grade ${data.severity_grade}).`);
            logToConsole('Gate 3 [Lesions]: Extracting Microaneurysm & Exudate topology...');
            gradeResult.innerText = `Grade ${data.severity_grade} (Referable)`;
            gradeResult.className = 'critical';
        }
        
        logToConsole('Gate 4 [Staging]: 5-Class ICDR calculation complete.');
        logToConsole('Rendering XAI Anatomical Overlays...');
        
        // Draw ETDRS Rings
        drawETDRSRings(data.fovea_x, data.fovea_y);
        
        foveaResult.innerText = `[${data.fovea_x.toFixed(2)}, ${data.fovea_y.toFixed(2)}]`;
        foveaResult.className = 'normal';
        latencyResult.innerText = `${latency} ms`;
        
    } catch (error) {
        logToConsole(`Error: ${error.message}`, true);
        gradeResult.innerText = 'Error';
        gradeResult.className = 'critical';
    }
    
    runBtn.disabled = false;
});

function drawETDRSRings(relX, relY) {
    const ctx = xaiCanvas.getContext('2d');
    const w = xaiCanvas.width;
    const h = xaiCanvas.height;
    
    ctx.clearRect(0, 0, w, h);
    
    const absX = relX * w;
    const absY = relY * h;
    
    const maxR = Math.min(w, h);
    const radii = [maxR * 0.1, maxR * 0.3, maxR * 0.45];
    const colors = ['rgba(0, 255, 0, 0.8)', 'rgba(255, 255, 0, 0.8)', 'rgba(255, 0, 0, 0.8)'];
    
    for (let i = 0; i < radii.length; i++) {
        ctx.beginPath();
        ctx.arc(absX, absY, radii[i], 0, 2 * Math.PI);
        ctx.strokeStyle = colors[i];
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    
    // Crosshair
    ctx.beginPath();
    ctx.moveTo(absX - 10, absY);
    ctx.lineTo(absX + 10, absY);
    ctx.moveTo(absX, absY - 10);
    ctx.lineTo(absX, absY + 10);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
}
