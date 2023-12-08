import torch
from PIL import Image

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transforms = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img