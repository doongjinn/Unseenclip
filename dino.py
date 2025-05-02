import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mmcv
from mmcv.runner import get_dist_info
from mmseg.datasets import build_dataset, build_dataloader, PIPELINES # Import PIPELINES
from mmseg.models import build_segmentor # Required by build_dataset potentially
import torchvision.transforms as T # For potential direct use if needed
import numpy as np
from tqdm import tqdm
import os
import argparse
import torch.hub
# --- Import the custom dataset class --- 
# Assuming the script is run from the root of the CLIP-RC project
# Adjust the path if necessary based on your execution directory
try:
    # Try relative import first (if script is inside the project structure)
    from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
except ImportError:
    # Fallback: Add project root to sys.path (less ideal but might work depending on setup)
    # import sys
    # import os
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjust based on script location
    # sys.path.insert(0, project_root)
    # from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
    # OR provide a direct import path if the module is installed or accessible
    print("Warning: Could not import ZeroPascalVOCDataset20 via relative path. Ensure the script is run from a location where this import works or adjust the import path.")
    pass

from PIL import Image

# --- Custom Pipeline Step for Modifying Image based on GT ---
@PIPELINES.register_module()
class ModifyImageByGT:
    """
    Modifies the input image based on the ground truth segmentation mask.
    Pixels belonging to seen classes are set to 0.
    Pixels belonging to unseen classes or ignore index are kept as is.
    """
    def __init__(self, seen_classes, ignore_index=255):
        self.seen_classes = set(seen_classes)
        self.ignore_index = ignore_index
        print(f"ModifyImageByGT initialized. Seen classes: {self.seen_classes}, Ignore index: {self.ignore_index}")

    def __call__(self, results):
        """
        Args:
            results (dict): The results dict from the data loading pipeline.
                Requires 'img' (np.ndarray HWC) and 'gt_semantic_seg' (np.ndarray HW).
        Returns:
            dict: The updated results dict with modified 'img'.
        """
        img = results['img']
        gt_seg = results['gt_semantic_seg']

        if img is None or gt_seg is None:
            print("Warning: 'img' or 'gt_semantic_seg' is None in ModifyImageByGT. Skipping modification.")
            return results

        # Ensure gt_seg is 2D (HW)
        if gt_seg.ndim == 3 and gt_seg.shape[-1] == 1:
            gt_seg = gt_seg.squeeze(-1)
        elif gt_seg.ndim != 2:
            print(f"Warning: Unexpected gt_semantic_seg shape {gt_seg.shape} in ModifyImageByGT. Skipping modification.")
            return results

        # Create a mask for pixels belonging to seen classes
        # Initialize mask to False
        seen_mask = np.zeros_like(gt_seg, dtype=bool)
        for cls_id in self.seen_classes:
            seen_mask |= (gt_seg == cls_id)

        # Also consider the ignore index - pixels to keep unchanged
        # We only modify pixels where seen_mask is True
        # Pixels where seen_mask is False (unseen or ignore) remain unchanged.

        # Apply the modification: set seen class pixels in the image to 0
        # Expand seen_mask to HWC format to match image channels
        seen_mask_3c = np.expand_dims(seen_mask, axis=-1)
        # Set pixels to 0 where seen_mask_3c is True
        img[seen_mask_3c.repeat(img.shape[-1], axis=-1)] = 0

        results['img'] = img
        # gt_semantic_seg is no longer needed after this step for feature extraction
        # but keep it for potential pipeline compatibility if other steps use it.
        # If memory is a concern, you could potentially remove it here:
        # del results['gt_semantic_seg']

        return results


# --- Helper Functions ---
def get_img_key(img_meta):
    """Extracts the base filename without extension from img_meta."""
    try:
        filename = img_meta.get('filename', img_meta.get('ori_filename', ''))
        if filename:
            return os.path.splitext(os.path.basename(filename))[0]
    except Exception as e:
        print(f"Warning: Could not get filename from img_meta: {img_meta}. Error: {e}")
    return None

# --- Main Extraction Logic ---
def extract_modified_dino_features(config_path, save_path, batch_size=16, num_workers=4, device='cuda'):
    """
    Extracts DINOv2 CLS tokens after modifying input images based on GT seg mask
    (seen class pixels set to 0).
    """

    # 1. Load Config
    cfg = mmcv.Config.fromfile(config_path)
    print(f"Loaded configuration from: {config_path}")

    # Use settings from the config
    img_norm_cfg = cfg.img_norm_cfg
    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(img_norm_cfg['std'], dtype=np.float32)
    to_rgb = img_norm_cfg.get('to_rgb', True)

    # Get seen classes from config
    if not hasattr(cfg, 'base_class'):
        print("Error: Configuration file must contain 'base_class' (list of seen classes).")
        return
    seen_classes = cfg.base_class
    ignore_index = cfg.get('ignore_index', 255) # Get ignore index or default to 255

    # 2. Define the pipeline for modified feature extraction
    #    Load Image & Annotations -> Modify Image -> Resize -> Normalize -> Format
    dino_input_resolution = 224 # Standard for ViT-B14 used in the original script
    print(f"Using DINOv2 input resolution: {dino_input_resolution}x{dino_input_resolution}")
    feature_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'), # <<< Load Segmentation Ground Truth
        dict(type='ModifyImageByGT', seen_classes=seen_classes, ignore_index=ignore_index), # <<< Modify Image Pixels
        dict(type='Resize', img_scale=(dino_input_resolution, dino_input_resolution), keep_ratio=False),
        dict(type='Normalize', mean=mean, std=std, to_rgb=to_rgb),
        dict(type='DefaultFormatBundle'), # Converts to Tensor, handles formatting
        dict(
            type='Collect',
            keys=['img'],   # Only need the modified image tensor now
            meta_keys=(     # Keep necessary metadata
                'filename', 'ori_filename', 'ori_shape',
                'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'
            )
        )
    ]

    # 3. Build Dataset using the training data split from the config
    train_dataset_cfg = cfg.data.train.copy()
    train_dataset_cfg['pipeline'] = feature_pipeline
    # Ensure test_mode is False if LoadAnnotations requires it (usually does)
    # Or adjust LoadAnnotations config if possible for test_mode=True
    # Let's assume test_mode=False is okay here as we need GT
    train_dataset_cfg['test_mode'] = False
    if 'dataset' in train_dataset_cfg: # Handle wrapped datasets
         train_dataset_cfg['dataset']['pipeline'] = feature_pipeline
         train_dataset_cfg['dataset']['test_mode'] = False

    dataset = build_dataset(train_dataset_cfg)
    print(f"Built dataset with {len(dataset)} images using config splits (and GT).")


    # 4. Build DataLoader
    rank, world_size = get_dist_info()
    distributed = world_size > 1
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        dist=distributed,
        shuffle=False,
        pin_memory=True,
    )
    print(f"Built DataLoader with batch size {batch_size} (Distributed: {distributed}).")


    # 5. Load DINOv2 Model
    print("Loading DINOv2 model (dinov2_vitb14)...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
    dinov2_model = dinov2_model.to(device)
    dinov2_model.eval()
    print("DINOv2 model loaded.")


    # 6. Feature Extraction Loop (operates on modified images)
    features_dict = {}
    print("Starting feature extraction from modified images...")
    with torch.no_grad():
        prog_bar = tqdm(dataloader, desc="Extracting DINOv2 features (modified img)") if rank == 0 else dataloader
        for i, data_batch in enumerate(prog_bar):
            # DefaultFormatBundle puts tensor in 'img' key, metas in 'img_metas'
            if 'img' not in data_batch or 'img_metas' not in data_batch:
                 print(f"Warning: Skipping batch {i}, 'img' or 'img_metas' not found in data_batch.")
                 continue

            # Handle DataContainer if necessary
            if hasattr(data_batch['img'], 'data'):
                 images = data_batch['img'].data # Potentially a list
                 if isinstance(images, list): images = images[0]
            else:
                 images = data_batch['img'] # Assume tensor

            if hasattr(data_batch['img_metas'], 'data'):
                img_metas = data_batch['img_metas'].data
                if isinstance(img_metas, list): img_metas = img_metas[0]
            else:
                img_metas = data_batch['img_metas']

            images = images.to(device)

            # Extract DINOv2 features (CLS token) from the modified image
            features = dinov2_model.forward_features(images)
            cls_token = features.get('x_norm_clstoken', None)
            if cls_token is None:
                 if isinstance(features, torch.Tensor) and features.ndim == 3:
                     cls_token = features[:, 0]
                 elif isinstance(features, dict) and 'x_prenorm' in features:
                      cls_token = features['x_prenorm'][:, 0]
                 else:
                     if rank == 0: print(f"Warning: Could not extract CLS token from DINOv2 output for batch {i}. Output type: {type(features)}")
                     continue

            # Store features in dict with filename as key
            for idx, meta in enumerate(img_metas):
                img_key = get_img_key(meta)
                if img_key and idx < cls_token.shape[0]:
                    features_dict[img_key] = cls_token[idx].cpu() # Store on CPU
                elif not img_key:
                     if rank == 0: print(f"Warning: Skipping image in batch {i}, index {idx} due to missing filename in meta.")

            if i % 50 == 0:
                torch.cuda.empty_cache()

    # Synchronization and Saving (Rank 0 only)
    if distributed:
        torch.distributed.barrier()

    if rank == 0:
        print(f"\nFinished extraction. Found features for {len(features_dict)} images (on rank 0).")
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(features_dict, save_path)
        print(f"Modified DINOv2 features saved successfully to: {save_path}")


# --- Argument Parsing and Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DINOv2 features from images modified based on GT segmentation masks (seen=0, unseen=original).")
    parser.add_argument('config', help='Path to the training configuration file (.py) containing dataset info and base_class')
    parser.add_argument(
        '--output',
        default='modified_dinov2_features_extracted.pth', # Changed default name
        help='Path to save the extracted features (.pth file)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16, # Might need to reduce if modification adds memory overhead
        help='Batch size per GPU for feature extraction'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of workers per GPU for data loading'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device to use for extraction (e.g., "cuda", "cuda:0", "cpu")'
    )
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # Handle distributed initialization
    if args.launcher != 'none':
         from mmcv.runner import init_dist
         init_dist(args.launcher, backend='nccl')

    # Device handling
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU.")
        args.device = 'cpu'
    elif 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        args.device = f'cuda:{args.local_rank}'

    extract_modified_dino_features( # Renamed function call
        config_path=args.config,
        save_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=args.device
    )