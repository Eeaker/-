
import time
import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from config import cfg
from model import make_model

def benchmark():
    # 1. Setup Dummy Config & Model
    print("Setting up model for benchmarking...")
    cfg.MODEL.DEVICE_ID = "0"
    cfg.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
    cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model (using num_classes=100 as dummy)
    model = make_model(cfg, num_class=100, camera_num=6, view_num=1)
    model.to(device)
    model.eval()

    # 2. Benchmark Feature Extraction (Constraint: < 40ms/img)
    B = 16 # Batch size
    dummy_input = torch.randn(B, 3, 256, 128).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    print("Benchmarking Feature Extraction...")
    start = time.time()
    iterations = 50
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_batch_time = (end - start) / iterations
    avg_img_time = avg_batch_time / B
    print(f"Feature Extraction Time: {avg_img_time*1000:.2f} ms/image (Limit: 40ms)")
    
    if avg_img_time * 1000 > 40:
        print("WARNING: Feature extraction TOO SLOW!")
    else:
        print("PASS: Feature extraction speed OK.")

    # 3. Benchmark Retrieval/Re-Ranking (Constraint: < 30ms/query)
    # Simulate Gallery=10000, Query=1
    # Re-ranking involves calculating dist matrix (Query x Gallery) and k-reciprocal
    
    # Check if we can import re-ranking function
    try:
        from utils.metrics import re_ranking
        print("Benchmarking Re-Ranking...")
        
        num_gallery = 2000 # Estimate gallery size for BallShow (small dataset?)
        # Dataset description says "bounding_box_test" (Gallery)
        # Assuming typical size. If huge, we assume standard gallery.
        
        q_feat = torch.randn(1, 768).cpu()
        g_feat = torch.randn(num_gallery, 768).cpu()
        
        q_cam = torch.randint(0, 6, (1,)).cpu().numpy()
        q_pid = torch.randint(0, 100, (1,)).cpu().numpy()
        g_cam = torch.randint(0, 6, (num_gallery,)).cpu().numpy()
        g_pid = torch.randint(0, 100, (num_gallery,)).cpu().numpy()
        
        start = time.time()
        # Note: re_ranking usually takes numpy (N, D)
        # We benchmark the calculation of distance
        distmat = re_ranking(q_feat, g_feat, k1=20, k2=6, lambda_value=0.3)
        end = time.time()
        
        print(f"Re-Ranking Time (Gallery={num_gallery}): {(end-start)*1000:.2f} ms (Limit: 30ms)")
        
    except ImportError:
        print("Could not import re_ranking from utils.metrics. Skipping re-ranking benchmark.")

if __name__ == "__main__":
    benchmark()
