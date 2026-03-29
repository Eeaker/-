import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

# Assuming the root of TransReID is the working directory
sys.path.append('.')
from model import make_model
from processor.processor import do_inference
from datasets import make_dataloader
from utils.logger import setup_logger

def load_network(cfg, model_path, num_class, camera_num, view_num):
    model = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=view_num)
    model.load_param(model_path)
    model.eval()
    return model

class DualEnsembleModel(nn.Module):
    def __init__(self, model1, model2, alpha=0.5):
        super(DualEnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.alpha = alpha

    def forward(self, img, cam_label=None, view_label=None):
        # The dataloader is configured for 384x128 (for model2).
        # Model 1 (from our ultra-stable 256x128 run) requires exactly 256x128 images.
        B, C, H, W = img.shape
        if H != 256 or W != 128:
            img_m1 = F.interpolate(img, size=(256, 128), mode='bilinear', align_corners=False)
        else:
            img_m1 = img
            
        feat1 = self.model1(img_m1, cam_label=cam_label, view_label=view_label)
        feat2 = self.model2(img, cam_label=cam_label, view_label=view_label) # Receives native 384x128
        
        # L2 normalize before ensemble to ensure equal contribution scale
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Weighted fusion (default 50-50)
        fused_feat = (self.alpha * feat1) + ((1.0 - self.alpha) * feat2)
        return fused_feat

def main():
    parser = argparse.ArgumentParser(description="Online Dual-Model Ensemble Test")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--weight1", default="", help="path to model 1 weights (e.g. Softmax ViT)", type=str, required=True)
    parser.add_argument("--weight2", default="", help="path to model 2 weights (e.g. CenterLoss ViT or ResNet)", type=str, required=True)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=False)
    logger.info("Running Online Dual-Model Ensemble Test")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # Note: Because the input sizes in the configs differ, we must override model1's internal expectation slightly 
    # if it uses the config, but `make_model` just builds the graph. However, the patch embedding grid is fixed.
    # To fix this, we load another config for model1.
    import copy
    cfg_m1 = copy.deepcopy(cfg)
    cfg_m1.defrost()
    cfg_m1.INPUT.SIZE_TRAIN = [256, 128]  # TransReID make_model statically reads SIZE_TRAIN to build position embeddings
    cfg_m1.INPUT.SIZE_TEST = [256, 128]
    cfg_m1.freeze()
    
    logger.info(f"Loading Model 1 from {args.weight1} (Expected Input: 256x128)")
    model1 = load_network(cfg_m1, args.weight1, num_classes, camera_num, view_num)
    
    # Load Model 2 
    logger.info(f"Loading Model 2 from {args.weight2} (Expected Input: 384x128)")
    model2 = load_network(cfg, args.weight2, num_classes, camera_num, view_num)

    ensemble_model = DualEnsembleModel(model1, model2, alpha=0.5)
    ensemble_model = ensemble_model.cuda()
    
    # Run the standard TransReID inference loop which now includes our MST and Dynamic AQE
    # because `ensemble_model(img)` returns the fused features which `processor.py` catches.
    logger.info("Starting Ensemble Inference with MST and AQE...")
    start_time = time.time()
    do_inference(cfg, ensemble_model, val_loader, num_query)
    end_time = time.time()
    
    logger.info(f"Total inference time: {end_time - start_time:.2f}s")
    
if __name__ == "__main__":
    main()
