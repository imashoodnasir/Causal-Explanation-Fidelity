from torchvision import transforms

def get_transforms(train: bool=True, size: int=224):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
