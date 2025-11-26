import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import time

#set the path for the dataset

DATASET_PATH = 'dataset2'
IMAGE_SIZE = (64, 64)  # the size for all images

#saving the image data
x =[]

#saving the labels
y=[]

# Read each category folder
for label, category in enumerate(os.listdir(DATASET_PATH)):
    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.isdir(category_path):
        continue

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        try:
            #we convert it to gray so that the measure will be the difference between the gray scale in each photo
            img = Image.open(img_path).convert('RGBA').convert('L')  # Convert to grayscale after converting to colored photo
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img).flatten()  # Flatten into 1D each photo has its own array with 4096 element
            x.append(img_array)
            y.append(label)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

x = np.array(x)
y = np.array(y)

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

##################################
# Create and train the Decision Tree

def decision_tree ():
    start = time.time()
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate results
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(f"Execution Time for decision tree model: {end - start:.2f} seconds")



def MLP ():
    start = time.time()
    #build the mlp and train stage
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)

    #test it using the test part of the data set

    y_pred = mlp.predict(X_test)

    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(f"Execution Time for MLP model: {end - start:.2f} seconds")


def naive_bayes():
    start = time.time()
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluate
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(f"Execution Time for naive bayes model: {end - start:.2f} seconds")


print("=== Decision Tree Results ===")

decision_tree()

print("\n=== Naive Bayes Results ===")
naive_bayes()

print("\n=== MLP Classifier Results ===")
MLP()

