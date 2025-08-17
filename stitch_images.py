# Nicholas Moreland

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from skimage.feature import SIFT
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp
from scipy.spatial.distance import cdist

# Detect Keypoints
def detect_keypoints(dst_img, src_img):
    detector1 = SIFT()
    detector1.detect_and_extract(dst_img)
    keypoints1 = detector1.keypoints
    descriptors1 = detector1.descriptors

    detector2 = SIFT()
    detector2.detect_and_extract(src_img)
    keypoints2 = detector2.keypoints
    descriptors2 = detector2.descriptors

    return (descriptors1, descriptors2, keypoints1, keypoints2)

# Keypoint Matching
def match_keypoints(descriptors1, descriptors2, cross_check):
    distances = cdist(descriptors1, descriptors2, 'euclidean')

    idx1 = np.arange(descriptors1.shape[0])
    idx2 = np.argmin(distances, axis=1)

    if cross_check:
        new_match = np.argmin(distances, axis=0)
        boolean_match = idx1 == new_match[idx2]
        idx1 = idx1[boolean_match]
        idx2 = idx2[boolean_match]

    matches = np.column_stack((idx1, idx2))
    return matches

# Plot Keypoint Matches
def plot_keypoint_matches(matches, img1, img2, keypoints1, keypoints2):
    dst = keypoints1[matches[:, 0]]
    src = keypoints2[matches[:, 1]]

    figure = plt.figure(figsize=(8, 4))
    axis1 = figure.add_subplot(121)
    axis2 = figure.add_subplot(122)
    axis1.imshow(img1, cmap='gray')
    axis2.imshow(img2, cmap='gray')

    for i in range(src.shape[0]):
        coordB = [dst[i, 1], dst[i, 0]]
        coordA = [src[i, 1], src[i, 0]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data", axesA=axis2, axesB=axis1, color="red")
        axis2.add_artist(con)
        axis1.plot(dst[i, 1], dst[i, 0], 'ro')
        axis2.plot(src[i, 1], src[i, 0], 'ro')

    plt.show()
    return (src, dst)

# Estimate Affine Matrix
# Given
#     keypoints1 – A set of keypoints in the source image.
#     keypoints2 – A set of matching keypoints in the destination image.
# Return
#     A 3x3 affine matrix.
def compute_affine_matrix(keypoints1, keypoints2):
    # Number of keypoints
    N = len(keypoints1)

    # Initialize matrices and sizes
    matrix_A = []
    vector_B = []
    new_size = (2 * len(keypoints1), 1)
    affine_size = (2, 3)
    affine_row = [0, 0, 1]

    # Populate matrix A and vector B
    for j in range(N):
        matrix_A.append([keypoints1[j][0], keypoints1[j][1], 1, 0, 0, 0])
        matrix_A.append([0, 0, 0, keypoints1[j][0], keypoints1[j][1], 1])

        if (j > len(keypoints2) - 1):
            vector_B.append(0)
            vector_B.append(0)
        else:
            vector_B.append(keypoints2[j][0])
            vector_B.append(keypoints2[j][1])

    # Convert to NumPy arrays
    matrix_A_np = np.array(matrix_A)
    vector_B_np = np.reshape(vector_B, new_size)

    # Transpose matrix A
    matrix_A_transpose = np.transpose(matrix_A_np)

    # Compute products and inverse
    product1 = np.dot(matrix_A_transpose, matrix_A_np)
    product1_inverse = np.linalg.inv(product1)
    product2 = np.dot(matrix_A_transpose, vector_B_np)

    # Compute affine matrix
    affine_matrix = np.dot(product1_inverse, product2)
    affine_matrix = np.reshape(affine_matrix, affine_size)
    affine_matrix = np.r_[affine_matrix, [affine_row]]

    return affine_matrix

# Estimate Projective Matrix
# Given
#     keypoints1 – A set of keypoints in the source image.
#     keypoints2 – A set of matching keypoints in the destination image.
# Return
#     A 3x3 projective matrix.
def compute_projective_matrix(keypoints1, keypoints2):
    # Number of keypoints
    N = len(keypoints1)
    
    # Initialize matrices and sizes
    matrix_A = []
    vector_B = []
    new_size = (2 * len(keypoints1), 1)
    projective_size = (3, 3)
    projective_row = [1]

    # Populate matrix A and vector B
    for j in range(N):
        if (j > len(keypoints2) - 1):
            matrix_A.append([0, 0, 0, 0, 0, 0, 0, 0])
            matrix_A.append([0, 0, 0, 0, 0, 0, 0, 0])
            vector_B.append(0)
            vector_B.append(0)
        else:
            matrix_A.append([keypoints1[j][0], keypoints1[j][1], 1, 0, 0, 0, -(keypoints1[j][0] * keypoints2[j][0]), -(keypoints1[j][1] * keypoints2[j][0])])
            matrix_A.append([0, 0, 0, keypoints1[j][0], keypoints1[j][1], 1, -(keypoints1[j][0] * keypoints2[j][1]), -(keypoints1[j][1] * keypoints2[j][1])])
            vector_B.append(keypoints2[j][0])
            vector_B.append(keypoints2[j][1])

    # Convert to NumPy arrays
    matrix_A_np = np.array(matrix_A)
    vector_B_np = np.reshape(vector_B, new_size)

    # Transpose matrix A
    matrix_A_transpose = np.transpose(matrix_A_np)

    # Compute products and inverse
    product1 = np.dot(matrix_A_transpose, matrix_A_np)
    product1_inverse = np.linalg.inv(product1)
    product2 = np.dot(matrix_A_transpose, vector_B_np)

    # Compute projective matrix
    projective_matrix = np.dot(product1_inverse, product2)
    projective_matrix = np.r_[projective_matrix, [projective_row]]
    projective_matrix = np.reshape(projective_matrix, projective_size)
    
    return projective_matrix

# Given:
#     data – A set of observations.
#     model – A model to explain the observed data points.
#     min_samples – The minimum number of data points to fit a model to.
#     threshold – The maximum distance between the data point and the model for the data point to be considered as an inlier.
#     max_iterations – The number of iterations to run the algorithm.
# Return:
#     A model that fits the data well (inliers).
def ransac(data, model_given, min_samples, threshold, max_iterations):
    # Initialize variables
    num_inliers = 0 
    bestErr = np.inf 
    bestInliers = []
    bestFit = None  
    bestFit = np.random.default_rng()
    nSamples = len(data[0])
    model = model_given()
    iterations = 0
    
    # RANSAC loop
    while iterations < max_iterations: 
        # Randomly select a subset of the data
        idxs_rnd = bestFit.choice(nSamples, min_samples, replace=False)
        samples = [d[idxs_rnd] for d in data]
        
        # Try to estimate the model
        model.estimate(*samples)

        # Compute errors and inliers
        errors = np.abs(model.residuals(*data))
        inliers = errors < threshold
        
        # Calculate the sum of errors
        sum_of_errors = errors.dot(errors)
        
        # Count the number of inliers
        total_inliers = np.count_nonzero(inliers)
        
        # Increment the number of iterations
        iterations += 1 
        
        if (total_inliers > num_inliers or (total_inliers == num_inliers and sum_of_errors < bestErr)):
            num_inliers = total_inliers
            bestErr = sum_of_errors
            bestInliers = inliers

    if any(bestInliers):
            
        dataInliers = [d[bestInliers] for d in data]
        model.estimate(*dataInliers)
        
    return model, bestInliers

# Testing
if __name__ == "__main__":
    # Load images
    dst_img_rgb = np.asarray(Image.open("./data/Rainier1.png"))
    src_img_rgb = np.asarray(Image.open("./data/Rainier2.png"))

    # Check if images are RGBA and convert to RGB
    if dst_img_rgb.shape[2] == 4:
        dst_img_rgb = rgba2rgb(dst_img_rgb)
    if src_img_rgb.shape[2] == 4:
        src_img_rgb = rgba2rgb(src_img_rgb)

    # Convert images to grayscale
    dst_img = rgb2gray(dst_img_rgb)
    src_img = rgb2gray(src_img_rgb)

    # Detect keypoints
    print('Computing keypoint detection...')
    (descriptors1, descriptors2, keypoints1, keypoints2) = detect_keypoints(dst_img, src_img)

    # Keypoint Matching
    print('Keypoint matching...')
    (matches) = match_keypoints(descriptors1, descriptors2, cross_check=True)

    # Plot keypoint matches
    print('Plotting keypoint matches...')
    (src, dst) = plot_keypoint_matches(matches, dst_img, src_img, keypoints1, keypoints2)

    # Estimate Affine Matrix
    print('Computing affine matrix...')
    affine_matrix = compute_affine_matrix(keypoints1, keypoints2)
    print(affine_matrix)

    # Estimate Projective Matrix
    print('Computing projective matrix...')
    projective_matrix = compute_projective_matrix(keypoints1, keypoints2)
    print(projective_matrix)

    # Estimate Affine Matrix using Similarity Transform model
    sk_M, sk_best = ransac((src[:, ::-1], dst[:, ::-1]), ProjectiveTransform, min_samples=4, threshold=1, max_iterations=300)
    print(sk_M)

    # Plot the best fit model
    print(np.count_nonzero(sk_best))
    src_best = keypoints2[matches[sk_best, 1]][:, ::-1]
    dst_best = keypoints1[matches[sk_best, 0]][:, ::-1]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img_rgb)
    ax2.imshow(src_img_rgb)

    for i in range(src_best.shape[0]):
        coordB = [dst_best[i, 0], dst_best[i, 1]]
        coordA = [src_best[i, 0], src_best[i, 1]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst_best[i, 0], dst_best[i, 1], 'ro')
        ax2.plot(src_best[i, 0], src_best[i, 1], 'ro')
    plt.show()

    # Transform the corners of img1 by the inverse of the best fit model
    rows, cols = dst_img.shape
    corners = np.array([
        [0, 0],
        [cols, 0],
        [0, rows],
        [cols, rows]
    ])

    corners_proj = sk_M(corners)
    all_corners = np.vstack((corners_proj[:, :2], corners[:, :2]))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1]).astype(int)
    print(output_shape)

    offset = SimilarityTransform(translation=-corner_min)
    dst_warped = warp(dst_img_rgb, offset.inverse, output_shape=output_shape)

    tf_img = warp(src_img_rgb, (sk_M + offset).inverse, output_shape=output_shape)

    # Combine the images
    foreground_pixels = tf_img[tf_img > 0]
    dst_warped[tf_img > 0] = tf_img[tf_img > 0]

    # Plot the result
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(dst_warped)
    plt.title('Stitched Image')
    plt.show()