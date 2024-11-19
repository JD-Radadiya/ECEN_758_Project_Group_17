# Image classification using Convolutional Neural Network (CNN) on the Fashion MNIST Dataset.

## Problem Overview
Fashion MNIST is a widely used benchmark that replaces the original MNIST datatset of handwritten digits. It comprises of 70,000 images; each image is 28x28 pixels in grayscale. 
In Fashion MNIST we have 10 classes and also the dificulty is increased by having images very similar to each other such as shirt and T-shirts in this new datatset. 
Fashion MNIST classes:
| Class | Label |
|:------:|:---------------------:|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Prerequisites
```bash
pip install -r requirements.txt
```
## Data Preprocessing 
Data preprocessing included normalization to scale the pixel values to range [-1,1], random rotation (up to 10 degrees), and random horizontal flipping with a 50% probability to enhance the generalizablity of the the model. 

## Model Architecture
The model architecture was implemented using PyTorch. It consisted of multiple convolutional layers followed by batch normalization, ReLU activations and max pooling layers to extract meaningful features from the images. The final fully connected layers are structured to generate output class probabilities through a softmax layer. <br>
The optimized model achieved a validation accuracy of 91% with a corresponding score of 0.89. We have used confusion matrix to support our analysis. 

![Confusion Matrix Validation](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/output_images/confusion_matrix_Validation.png)
![Confusion Matrix Test](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/output_images/confusion_matrix_Test.png)

To gain further insights into the learned representations of the model, we used UMAP a dimensionality reduction technique that helps visualize high-dimensional data in a two-dimensional space.

![UMAP Visualization](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/output_images/Embedding%20Clusters.png)

Hyperparameter tuning for the model was performed using Optuna, which resulted in an optimal learning rate of 0.001 and a batch size of 64. The training and validation loss and accuracy over epochs, as well as the average loss and accuracy across 5 folds for the training and validation data, were visualized and plotted to gain deeper insights.

![Loss Accuracy Plot](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/output_images/loss_accuracy_plot.png)
![Loss Accuracy Plot across 5 folds](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/output_images/average_loss_accuracy_plot_across_5_folds.png)


## Challenges and Limitations
The analysis of misclassified samples highlighted challenges in distinguishing between visually similar classes. This limitation suggests that additional techniques, such as advanced feature extraction or attention mechanisms, could be explored to further improve classification performance.
![Missclassified Images Test](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/output_images/misclassified_images_Test.png)
![Missclassified Images Validation](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/output_images/misclassified_images_Validation.png)
