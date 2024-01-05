import cv2
import numpy as np

def compute_homography(image1_path, image2_path):
    # Read the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Match descriptors using FLANN matcher
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Compute homography if enough matches are found
    if len(good_matches) > 4:
        # Extract location of good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        # Compute homography
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
        return H
    else:
        print("Not enough matches found to compute homography.")
        return None

# Example usage
homography_matrix = compute_homography('path_to_image1.jpg', 'path_to_image2.jpg')
print("Homography Matrix:")
print(homography_matrix)
