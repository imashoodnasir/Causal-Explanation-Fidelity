import os
from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset
from .transforms import get_transforms

class FolderDataset(Dataset):
    """Generic folder dataset.
    Expects directory structure:
      root/
        class_0/
            img1.jpg, ...
        class_1/
            ...
    """
    def __init__(self, root: str, split: str='train', img_size: int=224):
        assert split in ['train','val','test']
        self.root = os.path.join(root, split)
        self.classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.samples = []
        for c in self.classes:
            cdir = os.path.join(self.root, c)
            for fname in os.listdir(cdir):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    self.samples.append((os.path.join(cdir, fname), self.class_to_idx[c]))
        self.transform = get_transforms(train=(split=='train'), size=img_size)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        path, y = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, y
