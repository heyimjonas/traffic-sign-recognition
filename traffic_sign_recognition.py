import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from PIL import Image
import pandas as pd

class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file=None, transform=None):
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".ppm")]
        self.labels = None
        if csv_file:
            df = pd.read_csv(csv_file, sep=";")
            self.labels = {os.path.basename(row["Filename"]): row["ClassId"] for _, row in df.iterrows()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.transform(Image.open(img_path)) if self.transform else Image.open(img_path)
        return (image, self.labels[os.path.basename(img_path)]) if self.labels else image

def load_data(data_dir, test_csv=None, batch_size=64):
    transforms_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transforms_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, "Final_Training/Images"), transform=transforms_train)
    test_data = CustomDataset(os.path.join(data_dir, "Final_Test/Images"), test_csv, transforms_test)

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size),
        len(train_data),
        len(test_data)
    )

def train_model(model, device, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def test_model(model, device, test_loader, criterion):
    model.eval()
    correct = total = test_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    print(f"Test Results:\nLoss: {test_loss/len(test_loader):.4f}\n"
          f"Accuracy: {100.*correct/total:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to GTSRB data directory")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test labels CSV")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_model", type=str, default="traffic_sign_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficSignNet().to(device)
    
    train_loader, test_loader, train_size, test_size = load_data(
        args.data_dir, args.test_csv, args.batch_size
    )

    print(f"Dataset sizes - Training: {train_size}, Testing: {test_size}")
    train_model(model, device, train_loader, 
                torch.optim.Adam(model.parameters(), lr=0.001),
                nn.CrossEntropyLoss(), args.epochs)
    
    test_model(model, device, test_loader, nn.CrossEntropyLoss())
    torch.save(model.state_dict(), args.output_model)
