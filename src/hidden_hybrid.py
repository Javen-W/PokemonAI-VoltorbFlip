import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import pickle


# Load visible and hidden state data
visible_states = pd.read_csv('./training_data/visible_states.csv')
hidden_states = pd.read_csv('./training_data/hidden_states.csv')

visible_states.head()
hidden_states.head()


# Path to the zip file containing screenshots
screenshots_dir = "./training_data/screenshots/"

# Image preprocessing pipeline
image_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),  # Resize images to a fixed size
    transforms.ToTensor(),       # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])


# Custom dataset to handle both tabular data and screenshots
class VoltorbFlipDataset(Dataset):
    def __init__(self, visible_states, hidden_states, screenshots_dir):
        self.visible_states = visible_states.drop(columns=["state_index"]).values
        self.hidden_states = hidden_states.filter(like="tiles_").values 
        self.screenshots_dir = screenshots_dir
        self.state_indices = visible_states["state_index"].values

    def __len__(self):
        return len(self.visible_states)

    def __getitem__(self, idx):
        # Tabular data
        x_tabular = torch.tensor(self.visible_states[idx], dtype=torch.float32)
        
        # Target data (shift hidden states from 1–4 to 0–3)
        y = torch.tensor(self.hidden_states[idx]-1, dtype=torch.long) # Shift: 1–4 → 0–3

        # Load corresponding screenshot
        state_index = self.state_indices[idx]
        image_path = f"{self.screenshots_dir}/{state_index}.png"
        image = Image.open(image_path)
        x_image = image_transform(image)
        #print(f"Target Shape in Dataset: {y.shape}")

        return (x_tabular, x_image), y

# Train-test split
visible_train, visible_test, hidden_train, hidden_test = train_test_split(
    visible_states, hidden_states, test_size=0.2, random_state=42
)

# Create datasets and data loaders
train_dataset = VoltorbFlipDataset(visible_train, hidden_train, screenshots_dir)
test_dataset = VoltorbFlipDataset(visible_test, hidden_test, screenshots_dir)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the ResNet-18 Model with Modified Input Layer
class ModifiedResNet18(nn.Module):
    def __init__(self, output_size):
        super(ModifiedResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,  # Single-channel input (grayscale)
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Output layer for your task
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)

    def forward(self, x):
        return self.resnet(x)


# Define the hybrid model
class HybridModel(nn.Module):
    MODEL_PATH = "./weights/hidden_hybrid.pkl"

    def __init__(self, tabular_input_size, image_output_size, num_classes, num_tiles=25):
        super(HybridModel, self).__init__()
        # Tabular data branch
        self.fc1 = nn.Linear(tabular_input_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Image data branch (using the modified ResNet-18)
        self.cnn = ModifiedResNet18(image_output_size)
        
        # # Image data branch (using pretrained ResNet-18)
        # self.cnn = models.resnet18(pretrained=True)
        # self.cnn.fc = nn.Linear(self.cnn.fc.in_features, image_output_size)

        # Combined branch
        self.fc_combined = nn.Linear(64 + image_output_size, num_tiles * num_classes)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)  ###????? missing in response

    def forward(self, x_tabular, x_image):
        # Tabular data forward pass
        x_tabular = self.relu(self.fc1(x_tabular))
        x_tabular = self.relu(self.fc2(x_tabular))

        # Image data forward pass
        x_image = self.cnn(x_image)

        # Combine features
        x_combined = torch.cat((x_tabular, x_image), dim=1)
        x_combined = self.fc_combined(x_combined)

        # Debug output shape
        # print(" self.fc_combined.out_features // 25", self.fc_combined.out_features // 25) ####################
        # print(f"x_tabular shape: {x_tabular.shape}")
        # print(f"x_image shape: {x_image.shape}")
        # print(f"x_combined shape before reshaping: {x_combined.shape}")
        
        # Reshape to output predictions for each tile
        batch_size = x_tabular.size(0)
        x_combined = x_combined.view(batch_size, 25, -1)
        # print(f"Model Output Shape: {x_combined.shape}")   ############

        return x_combined #self.softmax(x_combined)

    def load_weights(self):
        self.load_state_dict(torch.load(self.MODEL_PATH, weights_only=True))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
tabular_input_size = visible_states.shape[1] - 1  # Exclude state_index
image_output_size = 64  # Output size from CNN branch
# output_size = 25  # Number of tiles
num_classes = 4
model = HybridModel(tabular_input_size, image_output_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function
def train(model, train_loader, criterion, optimizer, epochs=10,num_tiles=25):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for (inputs_tabular, inputs_image), targets in train_loader:

            # Move data to device
            inputs_tabular, inputs_image, targets = (
                inputs_tabular.to(device),
                inputs_image.to(device),
                targets.to(device),
            )
            
            optimizer.zero_grad()
            outputs = model(inputs_tabular, inputs_image)   # Shape: [batch_size, 25, num_classes]
            # print(f"Model Output Shape: {outputs.shape}") #############
            # print(f"Targets Shape: {targets.shape}") #############

            # Reshape outputs and targets for loss calculation
            batch_size, num_tiles, num_classes = outputs.shape
            # print(" num_tiles:", num_tiles)
            outputs = outputs.view(batch_size * num_tiles, num_classes)  # [batch_size * num_tiles, num_classes]
            # print(f"Outputs Shape After Reshaping: {outputs.shape}")   ##########
            targets = targets.view(-1)  # [batch_size * num_tiles]
            # print(f"Targets Shape After Reshaping: {targets.view(-1).shape}") #############

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs_tabular, inputs_image), targets in test_loader:

            # Move data to device
            inputs_tabular, inputs_image, targets = (
                inputs_tabular.to(device),
                inputs_image.to(device),
                targets.to(device),
            )
            # Forward pass
            outputs = model(inputs_tabular, inputs_image)
            _, predicted = torch.max(outputs, 2)  ###

            # Map predictions back to 1-4
            predicted = predicted + 1
            targets = targets + 1  # Optional: Shift targets back for comparison
            
            total += targets.numel() #size(0)
            correct += (predicted == targets).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")


# Save the model (ensure it's on the CPU)
def save_model_pickle(model, path):
    model_cpu = model.to("cpu")  # Move the model to CPU
    with open(path, "wb") as f:
        pickle.dump(model_cpu, f)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # Train and evaluate the model
    train(model, train_loader, criterion, optimizer, epochs=10)
    evaluate(model, test_loader)
    
    # Save the model (ensure portability to CPU)
    save_model_pickle(model, HybridModel.MODEL_PATH)
