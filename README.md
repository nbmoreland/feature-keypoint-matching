# Feature Keypoint Matching & Image Stitching

A computer vision project implementing SIFT-based feature extraction, image classification, and panoramic image stitching using Python and scikit-image.

## 🎯 Project Overview

This repository contains two main computer vision applications:

1. **Image Classification Pipeline**: Implements a bag-of-visual-words approach using SIFT features extracted from CIFAR-10 images, followed by SVM classification
2. **Image Stitching Pipeline**: Creates panoramic images by detecting and matching SIFT keypoints between multiple images, computing transformation matrices, and seamlessly blending them together

## 🚀 Features

- **SIFT Feature Extraction**: Scale-Invariant Feature Transform for robust keypoint detection
- **Bag-of-Visual-Words**: K-means clustering to create visual vocabularies from SIFT descriptors
- **TF-IDF Weighting**: Term frequency-inverse document frequency for better feature representation
- **RANSAC Algorithm**: Random Sample Consensus for robust transformation estimation
- **Affine & Projective Transformations**: Support for both transformation types in image stitching
- **Cross-Check Matching**: Bidirectional keypoint matching for improved accuracy

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- Sufficient RAM for image processing (minimum 4GB recommended)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/feature-keypoint-matching.git
cd feature-keypoint-matching
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install numpy scikit-image scikit-learn matplotlib pillow scipy tqdm
```

## 📁 Project Structure

```
feature-keypoint-matching/
│
├── feature_extraction.py   # CIFAR-10 feature extraction and classification
├── evaluate_sift.py        # SVM training and evaluation utilities
├── load_and_split.py       # Dataset loading and preprocessing
├── stitch_images.py        # Image stitching implementation
├── CLAUDE.md              # Claude AI assistant documentation
├── README.md              # This file
│
├── data/                  # Image data directory (create if needed)
│   ├── Rainier1.png      # Example image 1 for stitching
│   └── Rainier2.png      # Example image 2 for stitching
│
└── *.npz                  # Generated data files (after running scripts)
    ├── cifar10.npz        # Preprocessed CIFAR-10 dataset
    └── cifar10_sift.npz   # Extracted SIFT features
```

## 💻 Usage

### 1. CIFAR-10 Classification Pipeline

First, download and prepare the CIFAR-10 dataset:
```bash
python load_and_split.py
```

Then run the feature extraction and classification:
```bash
python feature_extraction.py
```

This will:
- Load CIFAR-10 images
- Convert to grayscale
- Extract SIFT features
- Build a visual vocabulary using K-means (50 clusters by default)
- Apply TF-IDF transformation
- Train an SVM classifier
- Output classification accuracy

### 2. Image Stitching Pipeline

Place your images in the `data/` directory, then run:
```bash
python stitch_images.py
```

The script expects `Rainier1.png` and `Rainier2.png` by default. To use different images, modify the file paths in `stitch_images.py`:
```python
dst_img_rgb = np.asarray(Image.open("./data/your_image1.png"))
src_img_rgb = np.asarray(Image.open("./data/your_image2.png"))
```

This will:
- Load and convert images to grayscale
- Detect SIFT keypoints
- Match keypoints between images
- Estimate transformation matrix using RANSAC
- Warp and blend images into a panorama
- Display the stitched result

### 3. Using Individual Components

#### Extract SIFT features from a custom image:
```python
from skimage.feature import SIFT
from skimage.color import rgb2gray
import numpy as np
from PIL import Image

# Load and convert image
img = np.asarray(Image.open("your_image.jpg"))
gray = rgb2gray(img)

# Extract SIFT features
sift = SIFT()
sift.detect_and_extract(gray)
keypoints = sift.keypoints
descriptors = sift.descriptors
```

#### Train SVM on custom features:
```python
from evaluate_sift import sift_svm, evaluate_sift

# Train model
model = sift_svm(X_train, y_train)

# Evaluate
accuracy = evaluate_sift(model, X_test, y_test)
```

## 🔬 Technical Details

### SIFT (Scale-Invariant Feature Transform)
- Detects keypoints invariant to scale, rotation, and illumination changes
- Generates 128-dimensional descriptors for each keypoint
- Robust to viewpoint changes and noise

### Bag-of-Visual-Words
1. Extract SIFT features from all training images
2. Cluster features using K-means (default k=50)
3. Create histograms of visual word occurrences
4. Apply TF-IDF weighting to normalize features
5. Train SVM classifier on weighted histograms

### RANSAC for Image Stitching
- **Minimum samples**: 4 points for projective transformation
- **Threshold**: 1 pixel for inlier classification  
- **Max iterations**: 300
- **Model types**: Supports both affine and projective transformations

### Transformation Matrices
- **Affine**: 6 degrees of freedom (rotation, translation, scale, shear)
- **Projective**: 8 degrees of freedom (includes perspective distortion)

## 📊 Performance Considerations

- SIFT extraction is computationally intensive; processing time scales with image size
- K-means clustering performance depends on the number of clusters and features
- RANSAC iterations can be adjusted for speed vs. accuracy trade-off
- Large images may require significant memory for stitching operations

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error during SIFT extraction**
   - Reduce image size or batch size
   - Process images sequentially instead of in parallel

2. **Poor stitching results**
   - Ensure images have sufficient overlap (>30% recommended)
   - Increase RANSAC iterations
   - Adjust inlier threshold

3. **Low classification accuracy**
   - Experiment with different k values for clustering
   - Try different SVM kernels (modify `LinearSVC` in `evaluate_sift.py`)
   - Ensure sufficient training data

4. **Import errors**
   - Verify all dependencies are installed
   - Check Python version compatibility

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Improvement

- Add support for other feature detectors (ORB, SURF, etc.)
- Implement multi-image stitching for full panoramas
- Add GPU acceleration for faster processing
- Create a command-line interface with argparse
- Add unit tests for core functions
- Implement adaptive parameter tuning

## 📚 References

- [SIFT Paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) - Lowe, D.G. (2004)
- [Bag-of-Words Model](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)
- [RANSAC Algorithm](https://en.wikipedia.org/wiki/Random_sample_consensus)
- [scikit-image Documentation](https://scikit-image.org/)

## 👤 Author

Nicholas Moreland

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- scikit-image team for SIFT implementation
- CIFAR-10 dataset creators
- OpenML for dataset hosting