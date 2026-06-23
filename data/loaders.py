"""Data loading utilities"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class MedicalDataset(Dataset):
    """Medical imaging dataset"""
    
    def __init__(self, image_paths, texts, labels, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.transform = transform or self._default_transform()
    
    @staticmethod
    def _default_transform():
        """Default image transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            image = torch.zeros(3, 224, 224)
        
        text = self.texts[idx]
        label = self.labels[idx]
        
        return image, text, label


def create_data_loaders(image_paths, texts, labels, batch_size=32, split=(0.7, 0.15, 0.15)):
    """Create train, val, test data loaders"""
    dataset = MedicalDataset(image_paths, texts, labels)
    
    train_size = int(len(dataset) * split[0])
    val_size = int(len(dataset) * split[1])
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
