from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF
import torch
import numpy as np
import random

class OxfordPetSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, size=(128, 128), augment=False):
        self.dataset = OxfordIIITPet(root=root, download=True, target_types='segmentation', split='trainval' if train else 'test')
        self.size = size
        self.augment = augment

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        if self.augment and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long) - 1

        return image, mask

    def __len__(self):
        return len(self.dataset)

