import torch
import numpy as np
from utils.metrics import euclidean_distance, eval_func, aqe_euclidean_distance
import argparse

def main():
    parser = argparse.ArgumentParser(description="Offline Feature Ensemble Evaluation")
    parser.add_argument("--feat1", type=str, required=True, help="Path to first model features (.pth)")
    parser.add_argument("--feat2", type=str, required=True, help="Path to second model features (.pth)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for the first model")
    parser.add_argument("--aqe", action="store_true", help="Apply Safe AQE on the ensemble features")
    args = parser.parse_args()

    print(f"Loading features from {args.feat1} and {args.feat2}...")
    # Expected format of saved dict: 
    # {'qf': qf, 'gf': gf, 'q_pids': q_pids, 'g_pids': g_pids, 'q_camids': q_camids, 'g_camids': g_camids}
    data1 = torch.load(args.feat1)
    data2 = torch.load(args.feat2)

    # Normalize before fusion
    qf1 = torch.nn.functional.normalize(data1['qf'].cuda(), dim=1, p=2)
    gf1 = torch.nn.functional.normalize(data1['gf'].cuda(), dim=1, p=2)
    
    qf2 = torch.nn.functional.normalize(data2['qf'].cuda(), dim=1, p=2)
    gf2 = torch.nn.functional.normalize(data2['gf'].cuda(), dim=1, p=2)

    # Late Fusion (Concatenation or Weighted Addition)
    print(f"Fusing features with alpha {args.alpha}...")
    qf_fused = args.alpha * qf1 + (1 - args.alpha) * qf2
    gf_fused = args.alpha * gf1 + (1 - args.alpha) * gf2

    # Re-normalize
    qf_fused = torch.nn.functional.normalize(qf_fused, dim=1, p=2)
    gf_fused = torch.nn.functional.normalize(gf_fused, dim=1, p=2)

    if args.aqe:
        print("Applying Safe AQE...")
        distmat = aqe_euclidean_distance(qf_fused, gf_fused, k=1, alpha=0.5, sim_thresh=0.8)
    else:
        distmat = euclidean_distance(qf_fused, gf_fused)

    print("Evaluating...")
    cmc, mAP = eval_func(distmat, data1['q_pids'], data1['g_pids'], data1['q_camids'], data1['g_camids'])
    
    print(f"Ensemble Results \n mAP: {mAP:.1%} \n Rank-1: {cmc[0]:.1%} \n Rank-5: {cmc[4]:.1%}")

if __name__ == '__main__':
    main()
