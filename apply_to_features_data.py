# apply_to_features_data.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from custom_neural_network import CustomNeuralNetwork

# Load dataset
df = pd.read_excel('features_data.xlsx')
df = df.drop(columns=['image_id'])

# Split dataset into features and target labels
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values.reshape(-1, 1)

# Split data into training and test sets
np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=3
)

# Standardize the data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Add bias term to inputs
X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

# Initialize and train the neural network
nn = CustomNeuralNetwork([X_train.shape[1], 32, 16, 1])
loss_history = nn.train(X_train, Y_train, epochs=1000, learning_rate=0.1)
nn.plot_loss(loss_history)

# Evaluate the model
Y_test_pred = nn.feedforward(X_test)
Y_pred_binary = (Y_test_pred >= 0.5).astype(int)

# Confusion matrix
confm = confusion_matrix(Y_test, Y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=confm)
disp.plot()
plt.show()

# Extract confusion matrix values
TN, FP, FN, TP = confm.ravel()

# Calculate metrics
accuracy = (TP + TN) / (TP + FP + FN + TN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f_score = 2 * ((precision * recall) / (precision + recall))

print("***************")
print(f"The accuracy of the classifier is {accuracy:.4f}")
print("***************")
print(f"The F1-score of the classifier is {f_score:.4f}")
print("***************")

# ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_test_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Find best threshold
gmeans = np.sqrt(tpr * (1 - fpr))
ix = np.argmax(gmeans)
best_threshold = thresholds[ix]

print("***************")
print(f"AUC: {roc_auc:.4f}")
print("***************")
print(f'Best Threshold = {best_threshold:.4f}, G-Mean = {gmeans[ix]:.4f}')
print("***************")
