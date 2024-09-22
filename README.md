# **Rotational Detection Using Contrastive Siamese Network**

## **Project Overview**

This project implements a **Rotational Detection System** using a **Siamese Network with Attention Mechanisms** and a **Contrastive Loss** function. The aim of the model is to differentiate between two images of the same object based on whether the images are rotated or not, particularly focusing on detecting if one image is rotated **180 degrees** relative to the other. The project utilizes **PyTorch** as the deep learning framework and includes data preprocessing, model training, and evaluation components.

The **dataloader** and **datagenerator** are specifically adapted to this task of rotational detection between pairs of images, determining whether one image is rotated by 180 degrees relative to another. The Siamese network architecture is well-suited for this task, comparing pairs of images and learning the rotational relationship between them.

The model can be trained on any image dataset, where the goal is to detect rotational transformations. The data is preprocessed to generate pairs of images (original and rotated), and the model is trained to minimize the contrastive loss between similar and dissimilar pairs.

With  little work of adaptation, this project could be used for other tasks, such as detection of similar images.
## **Key Features**

1. **Siamese Network with Attention**:
   - Uses a **Siamese Architecture** with a **Mobilenet_v2** backbone to process two images simultaneously.
   - Integrates an **Attention Module** to improve focus on important features of the images.
   
2. **Contrastive Loss Function**:
   - The model uses a **contrastive loss** to minimize the distance between features of similar pairs and maximize it for dissimilar pairs.

3. **Image Augmentation for Rotational Detection**:
   - Images are augmented by applying minor rotations and large rotations (close to 180 degrees) to simulate real-world scenarios.

4. **Detection of 180-Degree Rotation**:
   - The **dataloader** and **datagenerator** are designed to feed pairs of images into the model, with one image potentially rotated 180 degrees compared to the other. The Siamese network is trained to detect whether or not the second image in the pair is a rotated version of the first.

5. **ClearML Integration**:
   - Training and model monitoring are integrated with **ClearML**, allowing efficient task tracking, artifact management, and logging.

## **Technical Specifications**

### **1. Data Preprocessing and Augmentation**

The dataset is prepared by processing the images in the following steps:
- **Original Images**: Images are resized to a fixed dimension (448x448 pixels).
- **Augmented Images**: For each image, the following transformations are applied:
  - **Small Rotations** (between -0.5 and 0.5 degrees).
  - **Large Rotations** (close to 180 degrees).

These image pairs are then labeled as **0** (similar) for original and slightly rotated images and **1** (dissimilar) for original and highly rotated images.

### **2. Model Architecture**

The model is built using a **Siamese Network** architecture with the following components:
- **Backbone**: The base model is `mobilenet_v2`, which extracts features from the images.
- **Attention Module**: Focuses on the important features of the extracted image representations.
- **Contrastive Loss**: Calculates the loss based on the Euclidean distance between the two feature vectors.

### **3. Training**

The training process involves:
- **Dataset Preparation**: The dataset is created with pairs of original and rotated images, where one of the images may have a 180-degree rotation.
- **Training the Model**: The model is trained using the **Contrastive Loss** function, which helps the network learn to differentiate between similar and dissimilar image pairs, particularly focusing on detecting a 180-degree rotation.
- **Callbacks**: Early stopping and TensorBoard logging are used to track the training progress.

### **4. Evaluation**

During evaluation, the model compares pairs of images and calculates the Euclidean distance between their feature representations. If the distance is above a certain threshold, the model predicts that the images are dissimilar (rotated by 180 degrees), otherwise, they are considered similar (non-rotated).

### **5. Next steps**

After testing and using this model, we realized a flaw: when presented with two images that do not represent the same object, the model makes unpredictable predictions. Therefore, we need to add a third class for pairs of images that do not represent the same object. This way, the model will cover all possible cases.
