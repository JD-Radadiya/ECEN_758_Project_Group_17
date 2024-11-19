# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import optuna
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
from sklearn.model_selection import KFold
import os
import clip
import matplotlib.pyplot as plt

# %%
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# %%
# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,)),  # normalize the dataset to range [-1, 1]
    transforms.RandomRotation(10),  # rotate the image randomly by 10 degrees
    transforms.RandomHorizontalFlip() # flip the image horizontally with a 50% probability
    
])

transform_clip = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # normalize the dataset to range [-1, 1]
])

# Download and load the training data
trainset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)

# Create indices for training and validation splits
num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)

# Define split size
split = int(0.2 * num_train)  # 20% for validation
train_idx, val_idx = indices[split:], indices[:split]

# Create samplers for training and validation
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# Create data loaders with samplers
trainloader = DataLoader(trainset, batch_size=64, sampler=train_sampler)
valloader = DataLoader(trainset, batch_size=64, sampler=val_sampler)

# Download and load the test data
testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

testset_clip = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_clip)
testloader_clip = torch.utils.data.DataLoader(testset_clip, batch_size=64, shuffle=False)

# %%
# Let's check if our data is loaded correctly
images, labels = next(iter(trainloader))

print(images.shape)
print(len(trainloader))
print(labels.shape)
print(len(testloader))

# Check each sample in the dataset
missing_samples = any(img is None or lbl is None for img, lbl in trainset)

if missing_samples:
    print("Missing data detected in individual samples.")
else:
    print("No missing data found in individual samples.")

# %%
unique_labels, counts = np.unique(trainset.targets, return_counts=True)
print(f"Unique labels: {unique_labels}")
print(f"Counts: {counts}")

print("Number of samples per digit:")
for label, count in zip(unique_labels, counts):
    print(f"Digit {label}: {count}")

# Check if the data distribution is balanced
total_samples = len(trainset.targets)
min_samples = min(counts)
max_samples = max(counts)
balance_ratio = min_samples / max_samples

print(f"\nTotal samples: {total_samples}")
print(f"Balance ratio (min/max): {balance_ratio}")

# %%
# Let's plot all the unique labels from the dataset
# Define the text labels
text_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(text_labels[labels[idx].item()])
plt.savefig('./Images/output_image.png')
print("Unique labels plotted successfully")
plt.close()

# %%
class SimpleCNNImage(nn.Module):
    def __init__(self, dropout=0.45, kernel_size=3, num_classes=10, flag=True):
        super(SimpleCNNImage, self).__init__()

        # If flag is true, then we use grayscale images
        if flag:
            # First Convolutional Layer
            self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=1)
            self.bn1 = nn.BatchNorm2d(32)  # Batch normalization added
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization added
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)  # Batch normalization added
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)  # Batch normalization added
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Output Layer
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout1(self.relu3(self.bn3(self.fc1(x))))
        x = self.dropout2(self.relu4(self.bn4(self.fc2(x))))
        x1 = x
        x = self.fc3(x)
        return x1, x

# %%
# model architecture
model = SimpleCNNImage()
model.to(device)
print(model)

# %%
model_path = "./model/best_model_state.pth"
best_model = SimpleCNNImage(dropout=0.45).to(device)
criterion = nn.CrossEntropyLoss() 

# %%
def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=10):
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_losses.append(running_train_loss / len(train_loader))
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        running_valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                _, outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
                
        valid_losses.append(running_valid_loss / len(valid_loader))
        valid_accuracy = 100 * correct_valid / total_valid
        valid_accuracies.append(valid_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Valid Accuracy: {valid_accuracy:.2f}%")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./Images/loss_accuracy_plot.png')
    print("Loss and accuracy plot saved successfully")
    plt.close()

    return train_losses, valid_losses, train_accuracies, valid_accuracies

# %%
# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters to tune
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 5, 20)

    # Initialize the model, criterion, and optimizer with suggested parameters
    model = SimpleCNNImage(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_model(
        model, criterion, optimizer, trainloader, valloader, num_epochs
    )

    # Return the best validation accuracy (maximize)
    return max(valid_accuracies)

# %%
try:
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    best_model.load_state_dict(state_dict)
    best_model.eval()
    # extract the best parameters
    best_params = {
        'dropout': best_model.dropout,
        'learning_rate': best_model.learning_rate,
        'num_epochs': best_model.num_epochs
    }
except FileNotFoundError:
    # Create a study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # Print best parameters
    print("Best parameters found:")
    print(study.best_params)

    # Evaluate on the testing set with the best model
    best_params = study.best_params
    best_model = SimpleCNNImage(dropout=best_params['dropout']).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_model(
        best_model, criterion, optimizer, trainloader, valloader, best_params['num_epochs']
    )

    # Save the model state dict
    model_path = "./model/best_model_state.pth"
    torch.save(best_model.state_dict(), model_path)
    print(f"Model state dict saved to {model_path}")

# %%
# Plot the confusion matrix, f1 score, and classification report

def evaluate_model(model, loader):

    if loader == valloader:
        print("Evaluating on the validation set")
        flag = "val"
    elif loader == testloader:
        print("Evaluating on the test set")
        flag = "test"
        test_loss = 0
        correct_test = 0
        total_test = 0

    best_model.eval()
    y_true = []
    y_pred = []
    misclassified_images = []
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            _, outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Find misclassified images
            misclassified_mask = predicted != labels
            misclassified_images.extend(images[misclassified_mask])
            true_labels.extend(labels[misclassified_mask])
            predicted_labels.extend(predicted[misclassified_mask])

            if flag == "test":
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

    if flag == "test":
        test_accuracy = 100 * correct_test / total_test
        print(f"Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {test_accuracy:.2f}%")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return misclassified_images, true_labels, predicted_labels, y_true, y_pred

# Evaluate the model on the validation set
misclassified_images_val, true_labels_val, predicted_labels_val, y_true_val, y_pred_val = evaluate_model(best_model, valloader)

# Evaluate the model on the test set
misclassified_images_test, true_labels_test, predicted_labels_test, y_true_test, y_pred_test = evaluate_model(best_model, testloader)

# %%
# Define the number of folds
k_folds = 5

# Set up KFold
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store results for each fold
fold_train_losses = []
fold_valid_losses = []
fold_train_accuracies = []
fold_valid_accuracies = []

# Extract best parameters of the model
dropout = best_params['dropout']
learning_rate = best_params['learning_rate']
num_epochs = best_params['num_epochs']

# Loop over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(trainset)):
    print(f'Fold {fold + 1}/{k_folds}')
    
    # Create samplers for training and validation
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Create data loaders with samplers
    trainloader = DataLoader(trainset, batch_size=64, sampler=train_sampler)
    valloader = DataLoader(trainset, batch_size=64, sampler=val_sampler)

    # Initialize model, criterion, and optimizer
    model = SimpleCNNImage(dropout=dropout)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model for this fold
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_model(
        model, criterion, optimizer, trainloader, valloader, num_epochs=num_epochs
    )

    # Store the results
    fold_train_losses.append(train_losses)
    fold_valid_losses.append(valid_losses)
    fold_train_accuracies.append(train_accuracies)
    fold_valid_accuracies.append(valid_accuracies)

# Calculate the average performance across all folds
avg_train_losses = np.mean(fold_train_losses, axis=0)
avg_valid_losses = np.mean(fold_valid_losses, axis=0)
avg_train_accuracies = np.mean(fold_train_accuracies, axis=0)
avg_valid_accuracies = np.mean(fold_valid_accuracies, axis=0)

# Plotting average training and validation metrics across all folds
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(avg_train_losses, label='Training Loss')
plt.plot(avg_valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training and Validation Loss (Across Folds)')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(avg_train_accuracies, label='Training Accuracy')
plt.plot(avg_valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Average Training and Validation Accuracy (Across Folds)')
plt.legend()

plt.tight_layout()
plt.savefig('./Images/average_loss_accuracy_plot_across_5_folds.png')
print("Average loss and accuracy plot saved successfully")
plt.close()

# %%
# Plot confusion matrix, f1 score, and classification report
def plot_results(y_true, y_pred, name):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=text_labels, yticklabels=text_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix' + f" ({name})")
    plt.savefig(f'./Images/confusion_matrix_{name}.png')
    print(f"Confusion matrix saved successfully ({name})")
    plt.close()

    # F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"F1 Score: {f1:.4f}" + f" ({name})")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=text_labels)
    print("Classification Report:" + f" ({name})")
    print(report)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}" + f" ({name})")

# Plot the results for the validation set
plot_results(y_true_val, y_pred_val, "Validation")

# Plot the results for the test set
plot_results(y_true_test, y_pred_test, "Test")

# %%
# Function to visualize some of the misclassified images
def visualize_misclassified_images(misclassified_images, true_labels, predicted_labels, class_names, num_images=5, name=""):
    print(f"Misclassified images ({name})")
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(misclassified_images))):
        image = misclassified_images[i].cpu().numpy().squeeze()
        true_label = class_names[true_labels[i].cpu().item()]
        predicted_label = class_names[predicted_labels[i].cpu().item()]
        
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {true_label}\nPred: {predicted_label}')
        plt.axis('off')
    plt.savefig(f'./Images/misclassified_images_{name}.png')
    print(f"Misclassified images saved successfully ({name})")
    plt.close()

# Define class names for FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize some of the misclassified images
visualize_misclassified_images(misclassified_images_val, true_labels_val, predicted_labels_val, class_names, num_images=5, name="Validation")
visualize_misclassified_images(misclassified_images_test, true_labels_test, predicted_labels_val, class_names, num_images=5, name="Test")

# %% [markdown]
# **CLIP**

# %%
model_clip, preprocess = clip.load("ViT-B/32", download_root="/scratch/user/jdr/ECEN_758/model/", device=device)

# %%
#  Define class names for FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(testloader_clip, desc="Zer-Shot Classification with CLIP"):

        images = images.to(device)
        labels = labels.to(device)
        
        image_features = model_clip.encode_image(images)

        text_input = clip.tokenize(["a photo of a " + class_names[label] for label in labels]).to(device)
        text_features = model_clip.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        preds = similarity.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# %%
plot_results(all_labels, all_preds, "Zero-Shot Classification with CLIP")

# %%
# Implement text to image matching from a database of FashionMNIST images

def text_to_image_in_memory(text_query, loader, preprocess):
    # Encode the text input
    text_input = clip.tokenize(text_query).to(device)

    with torch.no_grad():
        text_features = model_clip.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    best_match = None
    best_similarity = -float('inf')

    for idx, (images, _) in tqdm(loader, desc="Text to Image Matching"):
        with torch.no_grad():
            for i in range(images.size(0)):  # Loop through each image in the batch
                image = preprocess(images[i]).unsqueeze(0).to(device)  # Preprocess individual images and add batch dimension
                image_features = model_clip.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                if similarity.max() > best_similarity:
                    best_similarity = similarity.max()
                    best_match = (idx, i)  # Track both batch and index within the batch

    
    return best_match, best_similarity

# %%
# Example multiple text query
text_queries = ["a photo of a sneaker", "a photo of a dress", "a photo of a t-shirt", "a photo of a bag", "a photo of a dog", "a photo of a cat"]
for text_query in text_queries:
    best_match, best_similarity = text_to_image_in_memory(text_query, testloader_clip, preprocess)
    print(f"Best match: {class_names[best_match]}, Similarity: {best_similarity.item():.4f}")