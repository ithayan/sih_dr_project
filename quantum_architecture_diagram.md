# Quantum Hybrid Pipeline Architecture

You can screenshot this diagram, or if you are using a standard Markdown viewer, right-click the diagram to export it as an image (PNG/SVG) for your presentation deck.

```mermaid
graph TD
    classDef input fill:#2b2d42,stroke:#8d99ae,stroke-width:2px,color:#edf2f4;
    classDef fno fill:#023047,stroke:#219ebc,stroke-width:2px,color:#8ecae6;
    classDef quantum fill:#38040e,stroke:#9d0208,stroke-width:2px,color:#ffba08;
    classDef aitac fill:#14213d,stroke:#fca311,stroke-width:2px,color:#e5e5e5;
    classDef output fill:#003049,stroke:#d62828,stroke-width:2px,color:#fcbf49;

    A["📸 Raw Retinal Fundus Image"]:::input --> B
    A --> C

    subgraph Geometry Branch [Spatial Vascular Mapping]
        B["🌐 Geo-FNO (Fourier Neural Operator)"]:::fno --> B1["🔄 FFT Latent Space Deformation"]:::fno
        B1 --> B2["Output: Continuous Vascular Geometry Mask"]:::fno
    end

    subgraph Quantum Lesion Branch [Feature Extraction]
        C["🧠 Classic CNN Feature Extractor"]:::quantum --> D["📉 Dimensionality Bottleneck (4D)"]:::quantum
        D --> E["⚛️ PennyLane Parameterized Quantum Circuit"]:::quantum
        E --> E1["🌀 Angle Embedding & Entanglement"]:::quantum
        E1 --> F["Output: Quantum Correlated Feature State"]:::quantum
    end

    F --> G["🧬 AI-TAC 1D CNN Sequencer"]:::aitac
    G --> H["🎯 Multi-Head Sequence Attention"]:::aitac
    
    B2 -. "Geometry used for spatial grounding (Optional routing)" .-> H
    
    H --> I["📊 Final Output: 5-Stage DR Severity Grade"]:::output
    
    %% Styling notes
    %% Using high-contrast tech themes suitable for dark-mode pitch decks
```

### Pro-Tip for the Pitch
When presenting this diagram:
1. Point to the **Geometry Branch** and explain that standard convolutions fail on tortuous vessels, which is why you mathematically map them using Fourier transforms.
2. Point to the **Quantum Lesion Branch** and explain how replacing thousands of classical parameters with just 4 entangled qubits is what allows this massive architecture to run on an offline clinic laptop in milliseconds.
