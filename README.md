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

The model architecture was implemented using PyTorch. It consists of multiple convolutional layers followed by batch normalization, ReLU activations, and max pooling layers to extract meaningful features from the images. The final fully connected layers are structured to generate output class probabilities through a softmax layer.


<div align="center">
  <b>Hierarchical Representation of CNN Architecture</b>
</div>

<div align="center">
  <img src="Images/model_arc.png" alt="Hierarchical Representation of CNN Architecture">
</div>



The output of the model is the classification of images into 10 classes:

<div align="center">
  <img src="Images/output_image.png" alt="Output Image">
</div>


The optimized model achieved:
- **Validation Accuracy**: 91.97%
- **F1-Score**: 0.9192

We used a confusion matrix to support our analysis:

<table border="1" align="center">
  <tr>
    <td align="center">
      <img src="Images/confusion_matrix_Validation.png" height="400">
      <br><b>Confusion Matrix - Validation</b><br>
    </td>
    <td align="center">
      <img src="Images/confusion_matrix_Test.png" height="400">
      <br><b>Confusion Matrix - Test</b><br>
    </td>
  </tr>
</table>


The highest accuracy was observed for classes like **Ankle Boot** and **Trouser**, while the model struggled slightly with distinguishing **Shirt** from **T-shirt** due to visual similarities.

To gain further insights into the learned representations of the model, we used **UMAP**, a dimensionality reduction technique that helps visualize high-dimensional data in a two-dimensional space.

### UMAP

![UMAP Visualization](Images/Embedding%20Clusters.png)
The plot shows that the embeddings of different classes are generally well-separated, indicating that the model was able to learn meaningful representations for each clothing item. However, some overlap was observed between similar classes, such as Shirt and T-shirt, which corresponds with the observed misclassification errors.

Hyperparameter tuning for the model was performed using Optuna, which resulted in an optimal learning rate of 0.00136, dropout of 0.4117 and num_epochs of 15. The training and validation loss and accuracy over epochs, as well as the average loss and accuracy across 5 folds for the training and validation data, were visualized and plotted to gain deeper insights.

### Loss Accuracy Plots
This section displays individual loss accuracy plots for different configurations of dropout, learning rate, and epochs. Each plot represents the performance of a model based on these parameters.
<table border="1" align="center">
  <tr>
    <td align="center">
      <img src="Images/Training and Validation Accuracy (dropout=0.5060569962417711,learning_rate=0.0015131419357681817, num_epochs=15).png" height="150">
      <br>Dropout: 0.5060569962417711<br>Learning Rate: 0.0015131419357681817<br>Epochs: 15
    </td>
    <td align="center">
      <img src="Images/Training and Validation Accuracy(dropout= 0.57438822948197, learning_rate= 0.0002734665436370785, num_epochs= 20).png" height="150">
      <br>Dropout: 0.57438822948197<br>Learning Rate: 0.0002734665436370785<br>Epochs: 20
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Images/Training and Validation Accuracy(dropout=0.302662376503097, learning_rate=0.0006711681032783274, num_epochs=10).png" height="150">
      <br>Dropout: 0.302662376503097<br>Learning Rate: 0.0006711681032783274<br>Epochs: 10
    </td>
    <td align="center">
      <img src="Images/Training and Validation Accuracy(dropout=0.31304489896446114, learning_rate=0.003028274410083483, num_epochs= 6).png" height="150">
      <br>Dropout: 0.31304489896446114<br>Learning Rate: 0.003028274410083483<br>Epochs: 6
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Images/Training and Validation Accuracy(dropout=0.41174583224196415, learning_rate= 0.0013661846989760332, num_epochs=15).png" height="150">
      <br>Dropout: 0.41174583224196415<br>Learning Rate: 0.0013661846989760332<br>Epochs: 15
    </td>
    <td align="center">
      <img src="Images/Training and Validation Accuracy(dropout=0.5744934769555154, learning_rate=0.0002697910286824228, num_epochs=18).png" height="150">
      <br>Dropout: 0.5744934769555154<br>Learning Rate: 0.0002697910286824228<br>Epochs: 18
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Images/Training and Validation Loss (dropout = 0.42335724636463823, learning_rate= 0.0011355143791864917, num_epochs=8).png" height="150">
      <br>Dropout: 0.42335724636463823<br>Learning Rate: 0.0011355143791864917<br>Epochs: 8
    </td>
    <td align="center">
      <img src="Images/Training and Validation Loss(dropout= 0.46911563618513663, learning_rate=0.0066009679215182585, num_epochs=15).png" height="150">
      <br>Dropout: 0.46911563618513663<br>Learning Rate: 0.0066009679215182585<br>Epochs: 15
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Images/Training and Validation Loss(dropout=0.4163189202753613, learning_rate= 0.00012870762760288856, num_epochs=13).png" height="150">
      <br>Dropout: 0.4163189202753613<br>Learning Rate: 0.00012870762760288856<br>Epochs: 13
    </td>
    <td align="center">
      <img src="Images/Training and validation Accuracy(dropout=0.407685774541, learning_rate= 0.00041514816246175895, num_epochs=9).png" height="150">
      <br>Dropout: 0.407685774541<br>Learning Rate: 0.00041514816246175895<br>Epochs: 9
    </td>
  </tr>
</table>

### Loss Accuracy Plot across 5 Folds
This plot shows the average loss accuracy across five different training folds, providing a comprehensive view of the model's performance across multiple runs.
![Loss Accuracy Plot across 5 folds](Images/average_loss_accuracy_plot_across_5_folds.png)

## Image to Text Classification with CLIP
Results from the CLIP classification showed similar outcomes as the CNN results. For images that were visually similar their associated text label was misclassified, such as when coats were misclassified with pullovers 802 times by the model. Moreover, the accuracy and f1-score of the CLIP confusion matrix, 0.629 and 0.598, respectively, support the need for improving model pretraining.

<div align="center">
    <img src="Images/confusion_matrix_Zero-Shot Classification with CLIP.png" >
</div>


### CLIP Outputs 

<table border="1" align="center">
  <tr>
    <td align="center">
      <img src="Images/text_to_image_a_photo_of_a_bag.png" height="150">
      <br><b>Photo of Bag</b><br>
    </td>
    <td align="center">
      <img src="Images/text_to_image_a_photo_of_a_dress.png" height="150">
      <br><b>Photo of Dress</b><br>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Images/text_to_image_a_photo_of_a_sneaker.png" height="150">
      <br><b>Photo of Sneaker</b><br>
    </td>
    <td align="center">
      <img src="Images/text_to_image_a_photo_of_a_t-shirt.png" height="150">
      <br><b>Photo of T-shirt</b><br>
    </td>
  </tr>
</table>

## Challenges and Limitations

The analysis of misclassified samples highlighted challenges in distinguishing between visually similar classes. This limitation suggests that additional techniques, such as advanced feature extraction or attention mechanisms, could be explored to further improve classification performance.

<table border="1" align="center">
  <tr>
    <td align="center">
      <img src="Images/misclassified_images_Test.png" height="`30">
      <br><b>Misclassified Images Test</b><br>
    </td>
    <td align="center">
      <img src="Images/misclassified_images_Validation.png" height="`30">
      <br><b>Misclassified Images Validation</b><br>
    </td>
  </tr>
  <tr>
    
