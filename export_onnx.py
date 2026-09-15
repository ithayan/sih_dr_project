import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_gnn_kan_pipeline import NetraXPipeline

class NetraX_ONNXExportWrapper(nn.Module):
    """
    Export wrapper for NetraX Swin-GNN-KAN architecture.
    Provides clean static tensor tuple outputs compatible with MATLAB importONNXNetwork.
    
    Inputs:
      - fundus_scan: (1, 3, 224, 224) float32 in [0, 1]
      
    Outputs:
      1. severity_logits: (1, 5) float32 [ICDR Grades 0 to 4]
      2. vascular_biomarkers: (1, 3) float32 [AVR, Tortuosity Index, Branching Angle]
      3. quality_score: (1, 1) float32 [Gate 1 Quality Scorer]
      4. anomaly_prob: (1, 1) float32 [Gate 2 Referable Probability]
      5. anatomical_landmarks: (1, 4) float32 [Fovea (x, y), Optic Disc (x, y)]
    """
    def __init__(self, core_model):
        super().__init__()
        self.core = core_model

    def forward(self, x):
        # Forward pass through tripartite architecture
        out = self.core(x)
        
        logits = out["severity_logits"]               # (1, 5)
        
        # Flatten biomarker tensor to static (1, 3)
        avr = out["biomarkers"]["avr"].view(-1, 1)
        tort = out["biomarkers"]["tortuosity"].view(-1, 1)
        angle = out["biomarkers"]["branching_angle_deg"].view(-1, 1)
        biomarkers = torch.cat([avr, tort, angle], dim=-1) # (1, 3)
        
        quality = out["quality_score"].view(-1, 1)    # (1, 1)
        anomaly = out["anomaly_prob"].view(-1, 1)     # (1, 1)
        
        landmarks = torch.cat([out["fovea_coords"], out["optic_disc_coords"]], dim=-1) # (1, 4)
        
        return logits, biomarkers, quality, anomaly, landmarks


def export_netrax_onnx(output_path="netrax_swin_gnn_kan.onnx"):
    print("================================================================================")
    print("  NETRAX: STATIC SHAPE ONNX EXPORT ENGINE FOR MATLAB IMPORT")
    print("  Target Model: Hybrid Swin-GNN-KAN Pipeline")
    print("  Deployment Target: Standard Edge Workstation (Intel Core i5, Offline)")
    print("================================================================================")

    model = NetraXPipeline()
    model.eval()

    export_wrapper = NetraX_ONNXExportWrapper(model)
    export_wrapper.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print("\n1. Verifying PyTorch Forward Pass...")
    with torch.no_grad():
        logits, biomarkers, quality, anomaly, landmarks = export_wrapper(dummy_input)

    print(f"   - Severity Logits Shape:       {tuple(logits.shape)}")
    print(f"   - Vascular Biomarkers Shape:   {tuple(biomarkers.shape)}")
    print(f"   - Quality Score Shape:         {tuple(quality.shape)} (Score: {quality.item():.3f})")
    print(f"   - Anomaly Prob Shape:          {tuple(anomaly.shape)}")
    print(f"   - Landmarks Shape:             {tuple(landmarks.shape)}")

    print(f"\n2. Exporting Static ONNX Model to: {output_path}...")
    torch.onnx.export(
        export_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['fundus_scan'],
        output_names=[
            'severity_logits',
            'vascular_biomarkers',
            'quality_score',
            'anomaly_prob',
            'anatomical_landmarks'
        ],
        dynamic_axes=None  # Explicitly static shapes for maximum MATLAB/C++ inference speed
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] Export Complete: {output_path} ({file_size_mb:.2f} MB)")
    print("\nMATLAB Ingestion Syntax:")
    print(f"  net = importONNXNetwork('{output_path}', 'OutputDataFormats', 'BC');")
    print("================================================================================")


if __name__ == "__main__":
    export_netrax_onnx()
