import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path

class MarkingDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Dataset for loading name images with different marking types
        
        Parameters:
        root_dirs (dict): Dictionary mapping class names to directory paths
        transform: Optional transform to be applied to images
        """
        self.image_paths = []
        self.labels = []
        
        # Create a mapping of classes to numerical labels
        self.class_to_idx = {
            'unmarked': 0,
            'sidemarked': 1,
            'namemarked': 2
        }
        
        # Collect all image paths and their corresponding labels
        for class_name, dir_path in root_dirs.items():
            class_idx = self.class_to_idx[class_name]
            for img_path in Path(dir_path).glob('*'):  # Adjust pattern if needed for specific extensions
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:  # Add more extensions if needed
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
        
        self.transform = transform if transform else transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MarkingDetector(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(MarkingDetector, self).__init__()
        # Use ResNet18 as base model
        self.base_model = models.resnet18(pretrained=pretrained)
        
        # Modify the final layer for our classification task
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cuda'):
    """
    Train the marking detection model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
    
    return best_model_state

def prepare_data(data_dirs, batch_size=32):
    """
    Prepare data loaders for training and validation
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = MarkingDataset(data_dirs, transform=transform)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define data directories with base path
    base_path = 'marked_model/data'
    data_dirs = {
        'unmarked': os.path.join(base_path, 'unmarked'),
        'sidemarked': os.path.join(base_path, 'sidemarked'),
        'namemarked': os.path.join(base_path, 'namemarked')
    }
    
    # Prepare data
    train_loader, val_loader = prepare_data(data_dirs)
    
    # Initialize model
    model = MarkingDetector(num_classes=3, pretrained=True)
    
    # Train model
    best_model_state = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=15,
        learning_rate=0.001,
        device=device
    )
    
    # Save the best model
    torch.save(best_model_state, 'best_marking_detector.pth')

if __name__ == '__main__':
    main()