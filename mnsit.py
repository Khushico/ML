import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Load the MNIST dataset
print("Loading the MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)

# Downsample for faster training (only 5000 samples)
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=5000, stratify=y, random_state=42)

# Normalize the pixel values
X_sample /= 255.0

# Handle NaN values by replacing them with 0
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_sample = imputer.fit_transform(X_sample)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

# Convert y_test to a NumPy array for proper indexing
y_test = np.array(y_test)

# Train a non-linear SVM model with RBF kernel
print("Training the non-linear SVM model...")
svm_model = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize and predict random test images
print("\nPredicting test images...")

def display_predictions(n_images=5):
    plt.figure(figsize=(15, 4))
    indices = np.random.choice(len(X_test), n_images, replace=False)  # Random selection
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)  # Reshape to 28x28 pixels
        predicted_label = y_pred[idx]
        actual_label = y_test[idx]

        plt.subplot(1, n_images, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {predicted_label}, True: {actual_label}")
        plt.axis('off')

    plt.show()

# Show 5 random image predictions
display_predictions()
