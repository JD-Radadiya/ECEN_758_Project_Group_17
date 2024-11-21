# Multi-Object Image Classification via Convolutional Neural Network on MNIST dataset

## Problem Overview
Fashion MNIST is a widely used benchmark that replaces the original MNIST datatset of handwritten digits. It comprises of 70,000 images; each image is 28x28 pixels in grayscale. 
In Fashion MNIST we have 10 classes and also the difficulty is increased by having images very similar to each other such as shirt and T-shirts in this new datatset. 
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
The model architecture was implemented using PyTorch. It consists of multiple convolutional layers followed by batch normalization, ReLU activations and max pooling layers to extract meaningful features from the images. The final fully connected layers are structured to generate output class probabilities through a softmax layer. <br>

<div align="center">
  <b>Hierarchical Representation of CNN Architecture</b>
</div> 

<div align="center">
    <img src="https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/Images/Hierarchical%20Representation%20of%20CNN%20Architecture.png" alt="Hierarchical Representation of CNN Architecture">
</div>

The optimized model achieved a validation accuracy of 91% with a corresponding F1-score of 0.89. We have used confusion matrix to support our analysis. 
![Confusion Matrix Validation](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/Images/confusion_matrix_Validation.png)
![Confusion Matrix Test](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/Images/confusion_matrix_Test.png)
The highest accuracy was observed for classes like Ankle Boot and Trouser, while the model struggled slightly with distinguishing Shirt from T-shirt due to visual similarities.

To gain further insights into the learned representations of the model, we used UMAP a dimensionality reduction technique that helps visualize high-dimensional data in a two-dimensional space.

![UMAP Visualization](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/Images/Embedding%20Clusters.png)
The plot shows that the embeddings of different classes are generally well-separated, indicating that the model was able to learn meaningful representations for each clothing item. However, some overlap was observed between similar classes, such as Shirt and T-shirt, which corresponds with the observed misclassification errors.

Hyperparameter tuning for the model was performed using Optuna, which resulted in an optimal learning rate of 0.001 and a batch size of 64. The training and validation loss and accuracy over epochs, as well as the average loss and accuracy across 5 folds for the training and validation data, were visualized and plotted to gain deeper insights.

# Loss Accuracy Plots

![Dropout: 0.2163, Learning Rate: 0.0007019, Epochs: 5](Images/loss_accuracy_plot_(dropout%3D0.21627798905909126%2C%20learning_rate%3D0.0007019114921215497%2C%20num_epochs%3D5).png)

Parameters:
- Dropout: 0.2163
- Learning Rate: 0.0007019
- Epochs: 5

![Dropout: 0.2807, Learning Rate: 0.0002627, Epochs: 11](images/loss_accuracy_plot_(dropout=0.2807492811706857,_learning_rate=0.0002627473501925258,_num_epochs=11).png)

Parameters:
- Dropout: 0.2807
- Learning Rate: 0.0002627
- Epochs: 11

![Dropout: 0.3485, Learning Rate: 0.0001965, Epochs: 15](images/loss_accuracy_plot_(dropout=0.3484810484584377,_learning_rate=0.0001965096091776,_num_epochs=15).png)

Parameters:
- Dropout: 0.3485
- Learning Rate: 0.0001965
- Epochs: 15

![Dropout: 0.3896, Learning Rate: 0.0003006, Epochs: 18](images/loss_accuracy_plot_(dropout=0.3896123874081374,_learning_rate=0.0003005946213353656,_num_epochs=18).png)

Parameters:
- Dropout: 0.3896
- Learning Rate: 0.0003006
- Epochs: 18

![Dropout: 0.3963, Learning Rate: 0.0002274, Epochs: 6](images/loss_accuracy_plot_(dropout=0.3963218764989253,_learning_rate=0.0002273976475531,_num_epochs=6).png)

Parameters:
- Dropout: 0.3963
- Learning Rate: 0.0002274
- Epochs: 6

![Dropout: 0.4174, Learning Rate: 0.0001271, Epochs: 8](images/loss_accuracy_plot_(dropout=0.4173550867945963,_learning_rate=0.0001271059770063,_num_epochs=8).png)

Parameters:
- Dropout: 0.4174
- Learning Rate: 0.0001271
- Epochs: 8

![Dropout: 0.45, Learning Rate: 0.001, Epochs: 18](images/loss_accuracy_plot_(dropout=0.45,_learning_rate=0.001,_num_epochs=18).png)

Parameters:
- Dropout: 0.45
- Learning Rate: 0.001
- Epochs: 18

![Dropout: 0.4731, Learning Rate: 0.0001203, Epochs: 12](images/loss_accuracy_plot_(dropout=0.4731226416454962,_learning_rate=0.0001202758743386732,_num_epochs=12).png)

Parameters:
- Dropout: 0.4731
- Learning Rate: 0.0001203
- Epochs: 12

![Dropout: 0.4999, Learning Rate: 0.0009574, Epochs: 14](images/loss_accuracy_plot_(dropout=0.499849001687473,_learning_rate=0.0009573848338439847,_num_epochs=14).png)

Parameters:
- Dropout: 0.4999
- Learning Rate: 0.0009574
- Epochs: 14

![Dropout: 0.5399, Learning Rate: 0.002960, Epochs: 18](images/loss_accuracy_plot_(dropout=0.53988785443921,_learning_rate=0.0029599474145136973,_num_epochs=18).png)

Parameters:
- Dropout: 0.5399
- Learning Rate: 0.002960
- Epochs: 18

![Dropout: 0.5761, Learning Rate: 0.0003109, Epochs: 10](images/loss_accuracy_plot_(dropout=0.5761434686873672,_learning_rate=0.0003108924781983657,_num_epochs=10).png)

Parameters:
- Dropout: 0.5761
- Learning Rate: 0.0003109
- Epochs: 10
![Loss Accuracy Plot across 5 folds](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/Images/average_loss_accuracy_plot_across_5_folds.png)

The output of the model is the classification of images into the 10 classes 
![Output Image](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/Images/output_image.png)


## Challenges and Limitations
The analysis of misclassified samples highlighted challenges in distinguishing between visually similar classes. This limitation suggests that additional techniques, such as advanced feature extraction or attention mechanisms, could be explored to further improve classification performance.<br>

<div align="center">
  <b>Missclassified Images Test</b>
</div> 

![Missclassified Images Test](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/Images/misclassified_images_Test.png)

<div align="center">
  <b>Missclassified Images Validation</b>
</div> 

![Missclassified Images Validation](https://github.com/JD-Radadiya/ECEN_758_Project_Group_17/blob/main/Images/misclassified_images_Validation.png)
