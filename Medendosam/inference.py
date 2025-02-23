import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from segment_anything import sam_model_registry
from dataset import Endovis18Dataset, VocalfolodsDataset
from model import Prototype_Prompt_Encoder, Learnable_Prototypes
from model_forward import model_forward_function
from matplotlib.colors import ListedColormap
import argparse
from utils import read_gt_endovis_masks, create_binary_masks, create_endovis_masks, eval_endovis

import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_and_save_colored_masks(endovis_masks, save_dir='./endovis_masks_visualizations_1'):
    """
    Visualize and save EndoVis masks with different colors for each class based on the given color map.

    Args:
        endovis_masks (dict): Dictionary containing EndoVis masks with keys as mask names.
                              Each mask should be a numpy array or a torch.Tensor.
        save_dir (str): Directory where the visualizations will be saved.
    """
    # Ensure the base save directory exists
    os.makedirs(save_dir, exist_ok=True)

    color_map = {
    0: (128, 128, 128),  # Void (Gray)
    1: (255, 0, 0),      # Vocal folds (Red)
    2: (0, 139, 205),    # Other tissue (Darker Deep Sky Blue)
    3: (0, 180, 0),      # Glottal space (Green)
    4: (128, 0, 128),    # Pathology (Purple)
    5: (255, 165, 0),    # Surgical tool (Orange)
    6: (255, 255, 0),    # Intubation (Yellow)
    7: (255, 105, 180)   # New class (Hot Pink)
}



    for mask_name, mask in endovis_masks.items():
        try:
            # Convert mask to numpy array if it's a torch.Tensor
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            # Ensure mask is a 2D array
            if mask.ndim != 2:
                raise ValueError(f"Mask {mask_name} is not a 2D array, but has shape {mask.shape}.")

            # Debug: Print unique values in the mask
            unique_values = np.unique(mask)
            print(f"Processing Mask: {mask_name}, Unique Values: {unique_values}")

            # Create a color image using the color map
            color_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for label, color in color_map.items():
                color_image[mask == label] = color

            # Construct the save path
            save_path = os.path.join(save_dir, f"{mask_name}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure subdirectories exist

            # Save the colorized image
            plt.imsave(save_path, color_image)

        except Exception as e:
            print(f"Error processing mask {mask_name}: {e}")

    print(f"Colored EndoVis masks saved in {save_dir}")



# Process arguments
print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2018", choices=["endovis_2018", "vocalfolds"], help='specify dataset')
parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3], help='specify fold number for endovis_2017 dataset')
args = parser.parse_args()

# Set parameters for inference
print("======> Set Parameters for Inference")
dataset_name = args.dataset
fold = args.fold
thr = 0
data_root_dir = f"../data/{dataset_name}"

# Load dataset-specific parameters
print("======> Load Dataset-Specific Parameters")
if "18" in dataset_name:
    num_tokens = 2
    dataset = Endovis18Dataset(data_root_dir=data_root_dir, mode="val", vit_mode="h")
    medendosam_ckp = "./work_dirs/endovis_2018/model_ckp.pth"
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir, mode="val")

elif "vocalfolds" in dataset_name:
    num_tokens=2
    dataset= VocalfolodsDataset(data_root_dir = data_root_dir, 
                                   mode="val",
                                   vit_mode = "h")
    medendosam_ckp = "./work_dirs/vocalfolds/model_ckp.pth"
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir = data_root_dir, mode = "val")


dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Load SAM
print("======> Load SAM")
sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
model_type = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()

# Load prototypes and prototype-based prompt encoder
print("======> Load Prototypes and Prototype-based Prompt Encoder")
learnable_prototypes_model = Learnable_Prototypes(num_classes=7, feat_dim=256).cuda()
protoype_prompt_encoder = Prototype_Prompt_Encoder(feat_dim=256, hidden_dim_dense=128, hidden_dim_sparse=128, size=64, num_tokens=num_tokens).cuda()

checkpoint = torch.load(medendosam_ckp)

protoype_prompt_encoder.load_state_dict(checkpoint['prototype_prompt_encoder_state_dict'])

sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])

# Set requires_grad to False
for model in [sam_prompt_encoder, sam_decoder, protoype_prompt_encoder, learnable_prototypes_model]:
    for param in model.parameters():
        param.requires_grad = False

# Start inference with EndoVis mask visualization
print("======> Start Inference with EndoVis Mask Visualization")
binary_masks = dict()
protoype_prompt_encoder.eval()
sam_decoder.eval()
learnable_prototypes_model.eval()

with torch.no_grad():
    prototypes = learnable_prototypes_model()

    for sam_feats, mask_names, cls_ids, _, _ in dataloader:
        sam_feats = sam_feats.cuda()
        cls_ids = cls_ids.cuda()
                
        preds, preds_quality = model_forward_function(protoype_prompt_encoder, sam_prompt_encoder, sam_decoder, sam_feats, prototypes, cls_ids)
        
        binary_masks = create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr)

    # Create EndoVis masks
    endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)

# Visualize and save EndoVis masks
visualize_and_save_colored_masks(endovis_masks)

# Evaluate results
endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)
print(endovis_results)
