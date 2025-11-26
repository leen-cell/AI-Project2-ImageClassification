
# Image Classification – Comparative Study (AI Project 2)

This project compares the performance of three machine learning models on an image classification task. The dataset includes three classes: **animals**, **flowers**, and **traffic signs**, with 800 images per class.

All images are converted to grayscale, resized to 64×64, flattened into 4096-element vectors, and split into training/testing sets (80/20).

For the full analysis, results, confusion matrices, and discussion, refer to the attached project report (PDF).

---

## Models Implemented

### Decision Tree

Built using scikit-learn’s `DecisionTreeClassifier`.
Simple, interpretable model that splits data based on pixel intensity thresholds.

### Naive Bayes

Implemented using `GaussianNB`.
Fastest model, suitable for high-dimensional continuous data like grayscale pixel values.

### Multi-Layer Perceptron (MLP)

Neural network with two hidden layers (256 and 128 neurons).
Trained for up to 300 iterations.
Provides the highest accuracy among the three models.

---

## Running the Code

### Requirements

```
numpy
pillow
scikit-learn
```

Install them with:

```bash
pip install numpy pillow scikit-learn
```

### Dataset Structure

```
dataset2/
    animals/
    flowers/
    traffic_signs/
```

### Run the project

```bash
python main.py
```

This will:

* Load and preprocess images
* Train Decision Tree, Naive Bayes, and MLP
* Print confusion matrices and classification reports
* Display execution time for each model

---

## Project Structure

```
Project2/
│── main.py
│── AiPproj2.pdf
│── README.md
│── dataset2
```

