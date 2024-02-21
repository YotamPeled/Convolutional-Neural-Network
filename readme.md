CIFAR-10 Classification with Convolutional Neural Networks (CNN) using PyTorch
Overview
This project focuses on implementing and training a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset, Using PyTorch, we build a CNN from scratch and also explore the concept of transfer learning by utilizing a pre-trained ResNet-18 model to improve our classification results.

Objectives
To understand and implement a CNN using PyTorch for image classification.
To learn how to work with the CIFAR-10 dataset.
To explore and practice the concept of transfer learning.
To fine-tune a pre-trained ResNet-18 model for our specific task.
Implementation Details
Custom CNN Architecture: The custom CNN model is designed specifically for the CIFAR-10 dataset. It includes several convolutional layers, pooling layers, and fully connected layers, along with appropriate activation functions.
Transfer Learning with ResNet-18: We leverage a pre-trained ResNet-18 model, available through PyTorch's torchvision.models, and fine-tune it for specific needs. This approach allows us to take advantage of the model's learned features from a much larger dataset (ImageNet).
Training and Evaluation: The models are trained using a GPU (if available) for faster computation. We track the training and validation loss and accuracy to monitor the models' performance.