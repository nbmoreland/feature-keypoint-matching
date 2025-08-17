# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computer vision project that implements feature extraction and image stitching using SIFT (Scale-Invariant Feature Transform) keypoints. The codebase has two main functionalities:

1. **CIFAR-10 Feature Extraction Pipeline**: Extracts SIFT features from CIFAR-10 images, builds a visual vocabulary using K-means clustering, and trains an SVM classifier for image classification.

2. **Image Stitching Pipeline**: Detects and matches SIFT keypoints between images, computes transformation matrices (affine/projective), and stitches images together using RANSAC for robust estimation.

## Architecture

### Core Components

**Feature Extraction System (`feature_extraction.py`)**
- Processes CIFAR-10 dataset images through SIFT feature extraction
- Implements bag-of-visual-words approach with K-means clustering (default: 50 clusters)
- Uses TF-IDF transformation to weight visual word histograms
- Trains LinearSVC model for classification
- Depends on pre-loaded data from `cifar10.npz`

**Image Stitching System (`stitch_images.py`)**
- Detects SIFT keypoints in grayscale images
- Matches keypoints using Euclidean distance with optional cross-checking
- Computes transformation matrices:
  - Affine transformation via least squares
  - Projective transformation via homography estimation
- Uses RANSAC for robust model fitting (4 samples, threshold=1, 300 iterations)
- Warps and combines images into panoramic output

**Supporting Modules**
- `evaluate_sift.py`: SVM training and evaluation utilities
- `load_and_split.py`: CIFAR-10 dataset fetching and preprocessing

### Data Flow

1. **Classification Pipeline**: 
   Raw images → Grayscale conversion → SIFT extraction → K-means clustering → Histogram encoding → TF-IDF → SVM training

2. **Stitching Pipeline**: 
   Two images → SIFT detection → Keypoint matching → RANSAC model fitting → Transformation estimation → Image warping → Stitched output

## Development Commands

### Running the Scripts

```bash
# Download and prepare CIFAR-10 dataset
python load_and_split.py

# Run feature extraction and classification pipeline
python feature_extraction.py

# Run image stitching (requires images in ./data/)
python stitch_images.py
```

### Dependencies

The project requires these key packages:
- scikit-image (SIFT feature detection)
- scikit-learn (KMeans, SVM, TF-IDF)
- numpy (array operations)
- matplotlib (visualization)
- PIL/Pillow (image loading)
- scipy (distance computations)
- tqdm (progress bars)

### Testing Individual Components

```python
# Test SIFT extraction on a single image
from skimage.feature import SIFT
from skimage.color import rgb2gray
sift = SIFT()
sift.detect_and_extract(grayscale_image)

# Test SVM evaluation
from evaluate_sift import sift_svm, evaluate_sift
model = sift_svm(X_train, y_train)
evaluate_sift(model, X_test, y_test)
```

## Key Implementation Details

- CIFAR-10 images are 32x32 RGB, stored in row-major order (R, G, B channels sequentially)
- SIFT descriptors are 128-dimensional vectors
- Cross-check matching ensures bidirectional consistency of keypoint matches
- RANSAC parameters are tuned for projective transformations with 4-point sampling
- Image stitching uses inverse warping to avoid holes in output

## Data Files

- `cifar10.npz`: Pre-split CIFAR-10 dataset (X_train, X_test, y_train, y_test)
- `cifar10_sift.npz`: Processed SIFT features and labels
- `./data/Rainier1.png`, `./data/Rainier2.png`: Example images for stitching demo