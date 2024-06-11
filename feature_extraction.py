# Nicholas Moreland
# 1001886051

from skimage.feature import SIFT
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from evaluate_sift import sift_svm, evaluate_sift
from sklearn.model_selection import train_test_split

# Import CIFAR-10 dataset
from sklearn.datasets import fetch_openml

# Extract SIFT features from an image
# Given
#     image: A grayscale image
# Returns
#     A list of SIFT features
def extract_features(image):
    sift = SIFT()
    sift_features = []
    y_features = []

    for idx in tqdm(range(image.shape[0]), desc="Processing images"):
        try:
            sift.detect_and_extract(image[idx])
            sift_features.append(sift.descriptors)
            y_features.append(image.target[idx]) # Only stores the label if the SIFT features are successfully extracted
        except:
            pass
    
    return sift_features, y_features

# Create a bag of visual words from a set of features
# Given
#     features: A list of SIFT features
#     k: The number of clusters (visual words)
# Returns
#   A set of visual words
def build_vocab(features, k):
    # Convert list of SIFT features to a single numpy array
    features_np = np.concatenate(features)

    # Create a KMeans model to cluster the SIFT features
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the KMeans model to the SIFT features
    kmeans.fit(features_np)

    # Return the cluster centers (visual words)
    return (kmeans, kmeans.cluster_centers_)


# Given
#     model: A KMeans model
#     features: A list of SIFT features
#     k: The number of clusters (visual words)
# Returns
#     A set of histograms representing the features
def encode_images(model, features, k):
    image_histograms = []

    for feature in tqdm(features, desc="Building histograms"):
        # Predict the closest cluster for each feature
        clusters = model.predict(feature)
        # Build a histogram of the clusters
        histogram, _ = np.histogram(clusters, bins=k, range=(0, k))
        image_histograms.append(histogram)

    # Convert the list of histograms to a numpy array
    image_histograms_np = np.array(image_histograms)
    return image_histograms_np

# Transform the image histograms using the TfidfTransformer
# Given
#     image_histograms_np: A set of histograms representing the features
# Returns
#     A set of histograms representing the features
def tfidf_transform(image_histograms_np):
    # Create a TfidfTransformer
    tfidf = TfidfTransformer()

    # Fit the TfidfTransformer to the image histograms
    tfidf.fit(image_histograms_np)

    # Transform the image histograms using the trained TfidfTransformer
    image_histograms_tfidf = tfidf.transform(image_histograms_np)

    return image_histograms_tfidf

if __name__ == "__main__":
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    # Extract the training and testing data
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Convert the input data to a numpy array. It is saved in row-major order where the first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    #rgb = np.array(X_train, dtype='uint8')
    rgb = np.concatenate([X_train, X_test], axis=0).astype('uint8')

    # Reshape the data to (num_images, height, width, num_channels)
    rgb = rgb.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Convert the images to grayscale
    gray = rgb2gray(rgb)

    # Visualize the first 10 images
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes):
        ax.imshow(gray[i], cmap='gray')
        ax.axis('off')
    plt.show()

    # Choose the number of clusters (visual words)
    num_clusters = 50

    # Extract SIFT features from the images
    sift_features, y_features = extract_features(gray)

    # Build a vocabulary of visual words
    (kmeans, kmeans.cluster_centers_) = build_vocab(sift_features, num_clusters)

    # Encode the images using the visual words
    image_histograms_np = encode_images(kmeans, sift_features, num_clusters)

    # Transform the image histograms using the TfidfTransformer
    image_histograms_tfidf = tfidf_transform(image_histograms_np)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(image_histograms_tfidf, np.array(y_features, dtype=int), test_size=0.2, random_state=42)

    # Train an SVM model using the SIFT features
    svm_model = sift_svm(X_train, y_train)

    # Evaluate accuracy of the SVM model
    evaluate_sift(svm_model, X_test, y_test)

    # Combine data into a dictionary
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    
    # Save the data to a file
    np.savez("cifar10_sift.npz", **data)