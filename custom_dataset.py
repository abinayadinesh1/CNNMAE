import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

from PIL import Image

from tqdm import tqdm

import random


# custom dataset used to load pairs for MAE training
class MaskedDataset(Dataset):
    def __init__(self, root_dir, img_dims, num_patches, batch_size):
        self.root_dir = root_dir

        self.convert_tensor = transforms.ToTensor()

        self.img_dims = img_dims

        # has to be a square of a number
        self.num_patches = num_patches
        self.patch_dim = int(img_dims[0]/np.sqrt(num_patches))

        self.batch_size = batch_size

        self.samples = []

        print("Loading dataset sample names...")

        # stores the file paths for every image in the dataset
        for sample in tqdm(os.listdir(root_dir)):
            self.samples += [root_dir + "/" + sample]


    def rescale(self, sample):
        return sample*2-1


    def __len__(self):
        return len(self.pairs)


    # loading the image from path
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # masking portion
        masked_patches = [random.randint(0, self.num_patches-1) for i in range(int(0.6*self.num_patches))]

        unmasked_image = Image.open(self.samples[idx]).resize(self.img_dims)

        # initialize masked image to be same as unmasked and iteratively zero out patches
        masked_image = self.convert_tensor(unmasked_image)

        # dim 0 is channels, dim 1 is y, dim 2 is x
        for patch_idx in masked_patches:
            y_start = int(patch_idx//np.sqrt(self.num_patches)*self.patch_dim)
            y_end = int((patch_idx//np.sqrt(self.num_patches)+1)*self.patch_dim)
            
            x_start = int(patch_idx%np.sqrt(self.num_patches)*self.patch_dim)
            x_end = int((patch_idx%np.sqrt(self.num_patches)+1)*self.patch_dim)

            masked_image[:,y_start:y_end,x_start:x_end] = 0

        rescaled_unmasked = self.rescale(self.convert_tensor(unmasked_image))
        rescaled_masked = self.rescale(masked_image)

        return {"masked": rescaled_masked.resize(self.batch_size, *rescaled_unmasked.size()), "unmasked": rescaled_unmasked.resize(self.batch_size, *rescaled_unmasked.size())}
        