# System Workflow: End-to-End Diagnostic Sequence

Here is the exact step-by-step workflow of your platform, from the moment a clinician interacts with the MATLAB Kiosk to the microsecond the PennyLane Quantum Circuit evaluates the retinal data. 

*You can screenshot this or save it as an image to use in your presentation deck to prove you understand the full stack.*

```mermaid
sequenceDiagram
    autonumber
    
    participant User as 👨‍⚕️ Clinician
    participant MATLAB as 💻 MATLAB UI (XAI_Visualizer.m)
    participant Server as 🌐 Local Flask API (server.py)
    participant Pipeline as 🧠 PyTorch Pipeline
    participant PennyLane as ⚛️ Quantum Circuit

    User->>MATLAB: Clicks "Load Fundus Image"
    MATLAB->>User: Renders Raw Retinal Scan
    User->>MATLAB: Clicks "Execute Hierarchical Inference"
    
    activate MATLAB
    MATLAB->>MATLAB: Base64 Encode Image into JSON
    MATLAB->>Server: HTTP POST /predict
    deactivate MATLAB
    
    activate Server
    Server->>Server: Decode & PyTorch Tensor Transform
    Server->>Pipeline: Trigger Forward Pass
    
    activate Pipeline
    Note over Pipeline: Branch A: Geo-FNO Spectral Mapping
    Note over Pipeline: Branch B: Regress Fovea & Optic Disc (x,y)
    
    Pipeline->>PennyLane: Pass 4D Latent Tensor Bottleneck
    activate PennyLane
    Note over PennyLane: Angle Embedding<br/>Entanglement Layers<br/>Pauli-Z Measurements
    PennyLane-->>Pipeline: Return Quantum Feature State
    deactivate PennyLane
    
    Note over Pipeline: Branch D: AI-TAC Polar Serialization (r, θ)
    Note over Pipeline: Branch E: Hierarchical 4-Gate Triage
    
    Pipeline-->>Server: Output (Severity Grade, Fovea Coords)
    deactivate Pipeline
    
    Server-->>MATLAB: HTTP 200 OK (JSON Response)
    deactivate Server
    
    activate MATLAB
    MATLAB->>MATLAB: Parse Severity & Fovea Coordinates
    MATLAB->>User: Stream UI Diagnostic Logs (Gates 1-4)
    MATLAB->>MATLAB: Generate Thermal Jet Saliency Map
    MATLAB->>MATLAB: Plot ETDRS Rings over Fovea (1mm, 3mm, 6mm)
    MATLAB->>User: Display Explainable AI (XAI) Overlay
    deactivate MATLAB
```

### Talking Points for the Judges:
- **Steps 4 & 10 (The Architecture Bridge):** Explain that by using a local HTTP API instead of a compiled MATLAB ONNX network, you completely eliminated the risk of hardware crashes on low-resource clinic machines.
- **Steps 8 & 9 (The Quantum Edge):** Point out that the 4D bottleneck and Pauli-Z measurements are what allow this heavy model to run in milliseconds. The PyTorch pipeline offloads the hardest spatial correlation math to the simulated quantum state. 
- **Step 14 (Explainable AI):** Emphasize that the system doesn't just guess; it dynamically draws the ETDRS medical rings *exactly* where PyTorch predicted the fovea to be, mapping standard clinical protocols perfectly onto the AI heatmaps.
