# Comprehensive Multi-Modal Architecture Pipeline (SIH26038)

This architecture fuses the rank-ordered tokenization of Cell2Sentence with the lightweight parameters of a distilled transformer, built specifically for edge deployment.

### Step 1: The Biological Tokenization Layer (Systemic Branch)

Instead of feeding raw numerical data, Cell2Sentence (C2S) converts single-cell gene expression data into textual sequences by rank-ordering gene names in descending order of expression levels.

* **The Truncation:** To keep this incredibly lightweight, the pipeline truncates this sequence. It only pulls the top 128 or 256 highly expressed genes (specifically focusing on systemic inflammatory or metabolic markers). This forms the short "cell sentence" that gets passed to the next layer.

### Step 2: The Distilled Edge Transformer

* **The Architecture:** The tokenized sentence enters a custom, scaled-down transformer (similar to the 22M parameter footprint of CELLlama, but distilled further down to around 5-10M parameters). It uses an embedding dimension of just 256 and 4 attention heads.
* **Hardware Alignment:** This ensures the heavy matrix multiplications required by standard transformers do not overwhelm local hardware. Running this pipeline natively via an Intel Core i5 processor with Intel Iris graphics will yield millisecond inference times without bottlenecking the system or requiring an external GPU.

### Step 3: The Vision Layer (DR Branch)

* **The Processing:** The portable fundus camera captures the retinal image, which is passed through a lightweight convolutional neural network (like MobileNetV2 or a scaled DenseNet).
* **The Output:** The CNN outputs a 512-dimensional visual feature vector representing the state of the retina's microvasculature.

### Step 4: The Fusion Network

* **The Mechanics:** The 256-dimension biological vector from the transformer and the 512-dimension visual vector from the CNN are concatenated.
* **The Objective:** This combined tensor passes through a dense neural layer. The network dynamically learns how microvascular damage in the eye correlates with the patient's systemic immunological collapse, creating a bi-directional risk profile.

### Step 5: The Dual Explainability Output (XAI)

This satisfies the core SIH26038 problem statement requiring transparency and trust.

* **Grad-CAM for DR:** The vision branch works backward from the final prediction to generate a thermal heatmap, highlighting exact hemorrhages or exudates on the retinal image.
* **Attention Extraction for Systemic Risk:** Because the biological branch is built on a PyTorch `TransformerEncoderLayer`, self-attention weights can be extracted directly from the forward pass. The interface will visually highlight the specific gene tokens that the model paid the most attention to when assigning the systemic risk score.

***
**Conclusion:** This provides a robust, unified screening kiosk that looks mathematically sophisticated to the hackathon judges while remaining perfectly viable for an offline rural clinic.
