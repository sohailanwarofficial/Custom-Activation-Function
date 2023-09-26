# Custom Activation Function and Decision Boundary Visualization

This repository contains code for demonstrating the use of a custom activation function in a neural network and visualizing decision boundaries.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- NumPy
- TensorFlow
- Keras
- Matplotlib
- Scikit-Learn (for generating training and testing data)

## Custom Activation Function

In this project, we implement and use a custom activation function named `TH`. The custom activation function is defined as follows:

```python
def TH(x):
    return tf.where(x >= 0, [1], [0])
```

We integrate this custom activation function into a Multi-Layer Perceptron (MLP) model for binary classification.

## Usage

1. Run the code in `custom_activation_function.py` to:

   - Generate training and testing datasets.
   - Create a neural network model using the custom activation function.
   - Train the model and evaluate its accuracy.

2. Run the code in `decision_boundary_visualization.py` to visualize the decision boundary of the model and compare it with a model using the ReLU activation function.

## Results

The code in `custom_activation_function.py` trains and evaluates the custom activation function model and prints the training and testing accuracy.

The code in `decision_boundary_visualization.py` visualizes the decision boundaries of the custom activation function model and a ReLU activation function model. It displays the training and testing points, allowing you to observe how the models classify points within and outside the specified boundary.

## Customization

You can customize the following aspects:

- Model architecture: Modify the model architecture in `custom_activation_function.py` and `decision_boundary_visualization.py` to experiment with different neural network structures.
- Number of training samples: Adjust the `train_samples` and `test_samples` variables in `custom_activation_function.py` to change the dataset size.
- Decision boundary shape: Modify the `boundary_points` variable in `custom_activation_function.py` to define a different boundary shape.

Feel free to modify and adapt the code to explore different custom activation functions and decision boundary visualizations.

---

Enjoy experimenting with custom activation functions and visualizing decision boundaries!
