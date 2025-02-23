import numpy as np
import torch
import cv2
import os
import os.path as osp
import re
from torch.utils.data import Dataset
from skimage.measure import label, regionprops

class VocalfolodsDataset(Dataset):
    def __init__(self, data_root_dir="../data/vocalfolds",
                 mode="val",
                 fold=0,
                 vit_mode="h",
                 version=0):
        self.vit_mode = vit_mode
        self.version = version
        self.mode = mode

        # directory containing all binary annotations
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
        elif mode == "val":
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
        # put all binary masks into a list
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue
            self.mask_list += [osp.join(osp.basename(subdir), i) for i in files]

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]

        # get class id from mask_name
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))

        # get pre-computed sam feature

        feat_dir = osp.join(
                self.mask_dir.replace(
                    "binary_annotations",
                    f"sam_features_{self.vit_mode}"
                ),
                mask_name.split("_")[0] + ".npy"
            )


        # get image path
        image_path = osp.join(
            self.mask_dir.replace(
                "binary_annotations",
                "images"
            ),
            mask_name.split("_")[0] + ".png"
        )
        # load SAM feature
        sam_feat = np.load(feat_dir)

        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # get class embedding
        class_embedding_path = osp.join(
            self.mask_dir.replace(
                "binary_annotations",
                f"class_embeddings_{self.vit_mode}"
            ),
            mask_name.replace("png", "npy")
        
        )
        class_embedding = np.load(class_embedding_path)
        return  sam_feat, mask_name, cls_id, mask, class_embedding

class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir="../data/endovis_2018", 
                 mode="val", 
                 vit_mode="h",
                 version=0):
        """
        EndoVis 2018 dataset class.

        Args:
            data_root_dir (str): Root directory containing the dataset.
            mode (str): One of "train", "val", or "test".
            vit_mode (str): ViT version of SAM, e.g., "h", "l", "b".
            version (int): Augmentation version (for train mode).
        """
        self.vit_mode = vit_mode
        self.version = version
        self.mode = mode

        # Define mask directory based on mode
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
        elif mode in ["val", "test"]:
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")
        else:
            raise ValueError(f"Invalid mode '{mode}', expected 'train', 'val', or 'test'.")

        # Store all binary masks
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue
            self.mask_list += [osp.join(osp.basename(subdir), i) for i in files]

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]

        # Extract class ID from filename
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))

        # Load precomputed SAM feature
        sam_feat_path = osp.join(
            self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"),
            mask_name.split("_")[0] + (".npy" if self.version == 0 else "npy.npy")
        )
        sam_feat = np.load(sam_feat_path)

        # Load ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Load class embedding
        class_embedding_path = osp.join(
            self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"),
            mask_name.replace("png", "npy")
        )
        class_embedding = np.load(class_embedding_path)

        return sam_feat, mask_name, cls_id, mask, class_embedding











