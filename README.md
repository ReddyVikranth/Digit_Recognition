🧠 Handwritten Digit Recognizer (from Scratch with NumPy)
This project implements a basic neural network from scratch using NumPy to classify handwritten digits (0–9) from the Kaggle Digit Recognizer dataset (MNIST subset). No high-level deep learning frameworks (like Keras or PyTorch) are used in model training — everything is manually built, including forward and backward propagation!

📁 Dataset
Source: Kaggle - Digit Recognizer

The dataset contains 28×28 grayscale images of digits.

Training set: 33,600 samples (80%)

Test set: 8,400 samples (20%)

📌 Features
Pure NumPy-based 2-layer neural network

Custom implementation of:

Parameter initialization

ReLU and Softmax activations

Forward propagation

Backpropagation (with one-hot encoded labels)

Gradient descent optimization

🏗️ Network Architecture
Input layer: 784 nodes (28×28 pixels)

Hidden layer: 28 nodes (ReLU activation)

Output layer: 10 nodes (Softmax activation)

🧪 Future Enhancements
Add a testing/prediction function with accuracy calculation

Introduce batch training and learning rate scheduling

Visualize training loss

Add support for custom input images

Authored by : G.Vikranth Reddy
