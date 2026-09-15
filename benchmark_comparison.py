import torch
import time
from classic_edge_pipeline import ClassicEdgePipeline
from quantum_hybrid_pipeline import QuantumHybridPipeline

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(model, model_name, dummy_input, iterations=10):
    print(f"\nBenchmarking {model_name}...")
    
    # Warm-up pass
    with torch.no_grad():
        _ = model(dummy_input)
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1000 # convert to ms
    
    params = count_parameters(model)
    
    print(f"--- {model_name} Results ---")
    print(f"Total Parameters: {params:,}")
    print(f"Average Latency:  {avg_latency:.2f} ms per image")
    
    return avg_latency, params

if __name__ == "__main__":
    print("Initializing Models (This may take a moment for Quantum libraries)...")
    
    classic_model = ClassicEdgePipeline()
    quantum_model = QuantumHybridPipeline()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("\nStarting Performance Comparison Suite")
    print("=" * 40)
    
    # Run classic for 10 iterations
    c_latency, c_params = benchmark_model(classic_model, "Classic Edge Pipeline", dummy_input, iterations=10)
    
    # Run quantum for only 2 iterations because CPU simulation is exponentially slow
    q_latency, q_params = benchmark_model(quantum_model, "Quantum Hybrid Pipeline", dummy_input, iterations=2)
    
    print("\n" + "=" * 40)
    print("FINAL COMPARISON")
    print("=" * 40)
    print(f"The Quantum model is {q_latency / c_latency:.1f}x SLOWER than the Classic Edge model.")
    if q_latency > 1000:
        print("CONCLUSION: The Quantum model exceeds 1 second (1000ms) per image, proving it is unviable for immediate offline edge deployment. The Classic Edge model is the clear winner for the rural kiosk.")
    else:
        print("CONCLUSION: Both models run in sub-second time on this hardware.")
