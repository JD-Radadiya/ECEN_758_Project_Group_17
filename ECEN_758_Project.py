# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
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

# %%
# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,)),  # normalize the dataset to range [-1, 1]
    transforms.RandomRotation(10),  # rotate the image randomly by 10 degrees
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()  # flip the image horizontally with a 50% probability
])

transform_clip = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),  # normalize the dataset to range [-1, 1]
    transforms.ToTensor(),
])

# Download and load the training data
trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

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

# Plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(text_labels[labels[idx].item()])
plt.show()

# %%
class SimpleCNNImage(nn.Module):
    def __init__(self, dropout=0.45, kernel_size=3, num_classes=10):
        super(SimpleCNNImage, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization added
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
print(model)

# %%
model_path = "./model/best_model_state.pth"
best_model = SimpleCNNImage(dropout=0.45)
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
    plt.show()
    
    return train_losses, valid_losses, train_accuracies, valid_accuracies

# %%
# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters to tune
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 5, 20)

    # Initialize the model, criterion, and optimizer with suggested parameters
    model = SimpleCNNImage(dropout=dropout)
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
    best_model.load_state_dict(torch.load(model_path))
    best_model.eval()
except:
    # Create a study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=7)

    # Print best parameters
    print("Best parameters found:")
    print(study.best_params)

    # Evaluate on the testing set with the best model
    best_params = study.best_params
    best_model = SimpleCNNImage(dropout=best_params['dropout'])
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
            _, outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

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
plt.show()

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
    plt.show()

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

    print("\nMost Common Misclassifications:")
    conf_df = pd.DataFrame(cm, index=text_labels, columns=text_labels)
    misclassified_pairs = conf_df.stack().reset_index()
    misclassified_pairs.columns = ["True Label", "Predicted Label", "Count"]
    misclassified_pairs = misclassified_pairs[misclassified_pairs["True Label"] != misclassified_pairs["Predicted Label"]]
    misclassified_pairs = misclassified_pairs.sort_values("Count", ascending=False)
    misclassified_pairs = misclassified_pairs.reset_index(drop=True)

    print(misclassified_pairs.head(10))
    print()    
    print()

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
    plt.show()

# Define class names for FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize some of the misclassified images
visualize_misclassified_images(misclassified_images_val, true_labels_val, predicted_labels_val, class_names, num_images=5, name="Validation")
visualize_misclassified_images(misclassified_images_test, true_labels_test, predicted_labels_val, class_names, num_images=5, name="Test")

# %% [markdown]
# 

# %% [markdown]
# **CLIP**

# %%
model_clip, preprocess = clip.load("ViT-B/32", device=device)

# %%
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(testloader_clip, desc="Zer-Shot Classification with CLIP"):

        images = images.to(device)
        labels = labels.to(device)
        
        image_features = model.encode_image(images)

        text_input = clip.tokenize(["a photo of a " + class_names[label] for label in labels]).to(device)
        text_features = model.encode_text(text_input)

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

    for idx, (images,_) in tqdm(loader, desc="Text to Image Matching"):
        with torch.no_grad():
            image_features = model_clip.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            if similarity.max() > best_similarity:
                best_similarity = similarity.max()
                best_match = idx
    
    return best_match, best_similarity

# %%
# Example text query
text_query = "a photo of a sneaker"
best_match, best_similarity = text_to_image_in_memory(text_query, testloader_clip, preprocess)
print(f"Best match: {class_names[best_match]}, Similarity: {best_similarity.item():.4f}")

# %%
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Function to extract embeddings
def extract_embeddings(model, image_train_loader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for image_data, image_labels in tqdm((image_train_loader), desc=f"Extracting"):
            image_input = image_data.view(-1, 1, 28, 28).float()
            image_embedding, _ = model(image_input)
            embeddings.append(image_embedding)
            labels.append(image_labels)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    return embeddings, labels

# Extract embeddings for training data
train_embeddings, train_labels = extract_embeddings(best_model, trainloader)

# %%
# Fit PCA
pca = PCA()
pca.fit(train_embeddings)

# Calculate cumulative explained variance ratio
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find the index where cumulative explained variance ratio first surpasses 0.9
index_90_percent = np.argmax(cumulative_variance >= 0.9)
print(f"Number of components for 90% explained variance: {index_90_percent}")

# Plot cumulative explained variance ratio
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Ratio')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Explained Variance')
plt.axvline(x=index_90_percent, color='g', linestyle='--', label='90% Variance Component')
plt.grid(True)
plt.legend()
# output_path = os.path.join(output_dir, 'Cumulative Explained Variance Ratio Image.png')
# plt.savefig(output_path)
plt.show()

# %%
# Function to apply PCA
def apply_pca(embeddings, n_components):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(embeddings)
    return pca_result

# Apply PCA
num_components = 8  # You can choose any number of components
pca_embeddings = apply_pca(train_embeddings, num_components)

# %%
print("pca_embeddings data type:", type(pca_embeddings))
print("pca_embeddings shape:", pca_embeddings.shape)

# Check the data type and shape of labels
train_labels_np = train_labels.numpy()
print("labels data type:", type(train_labels_np))
print("labels shape:", train_labels_np.shape)

# %%
import umap.umap_ as umap

# Function to apply UMAP, visualize embeddings, and perform k-means clustering
def visualize_umap_with_clustering(embeddings, labels, num_clusters):
    # Apply UMAP for dimensionality reduction
    umap_result = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    
    # Plot UMAP embeddings
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='tab10', s=3)
    plt.title('UMAP Visualization of Train Embeddings')
    plt.colorbar(label='Labels')
#     output_path = os.path.join(output_dir, 'UMAP Visualization of Train Embeddings Image.png')
#     plt.savefig(output_path)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(umap_result)
    cluster_centers = kmeans.cluster_centers_  # Get cluster centers
    
    # Plot UMAP embeddings with cluster assignments and cluster centers
    plt.subplot(1, 2, 2)
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=cluster_labels, cmap='viridis', s=3)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=50, label='Cluster Centers')
    plt.title('UMAP Visualization with K-means Clustering')
    plt.colorbar(label='Clusters')
    plt.legend()
#     output_path = os.path.join(output_dir, 'UMAP Visualization with K-means Clustering Image.png')
#     plt.savefig(output_path)
    plt.show()

# Assuming you have your embeddings stored in pca_embeddings and labels in train_labels_np
# Also, assuming the number of clusters is 10
visualize_umap_with_clustering(pca_embeddings, train_labels_np, num_clusters=10)


