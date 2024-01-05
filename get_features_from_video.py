import glob

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


import re
import time

def split_video_into_frames(video_path, output_folder, frame_rate=1):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Get video's frame rate
    video_fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval (if frame_rate is 1, we take 1 frame per second)
    frame_interval = int(video_fps / frame_rate)

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Save frames at specific interval
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_name, frame)
            print(f"Saved {frame_name}")

        frame_count += 1

    # Release the video capture object
    video.release()


def get_image_paths(folder_path, image_extensions=['jpg', 'jpeg', 'png', 'gif', 'bmp']):
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, f'*.{extension}')))
    return image_paths

def feature_extractor(image1_path, image2_path):
    # Read images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features in both images
    print("Detecting SIFT features...")
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    # Create feature matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    print("Matching features...")
    matches = bf.match(descriptors_1, descriptors_2)

    # Sort them in order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    matches = matches[:50]

    print(len(matches))

    print(matches)

    coordinates_image1 = []
    coordinates_image2 = []

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # Get the coordinates
        (x1, y1) = keypoints_1[img1_idx].pt
        (x2, y2) = keypoints_2[img2_idx].pt

        coordinates_image1.append((x1, y1))
        coordinates_image2.append((x2, y2))
    vecs = []
    for i, (coord1, coord2) in enumerate(zip(coordinates_image1, coordinates_image2)):
        # get vector between two coordinates

        vec = np.array(coord1) - np.array(coord2)
        print(vec)
        vecs.append(vec[0])
        
        print(f"Match {i}: Image 1 - {coord1}, Image 2 - {coord2}")

    sorted_vecs = sorted(vecs)
    median = sorted_vecs[len(sorted_vecs)//2]

    direction = median * 10
    print('direction', direction)

    # Draw first 50 matches
    matched_img = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Convert BGR to RGB for matplotlib display
    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    print("Displaying matches...")
    # Display the matches
    plt.imshow(matched_img)
    plt.show()

def compute_homography(image1_path, image2_path):
    # Read the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    keypoint_coordinates = [keypoint.pt for keypoint in keypoints1]


    # Match descriptors using FLANN matcher
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    plt.imshow(image1)
    plt.imshow(image2)
    plt.show()


    # Compute homography if enough matches are found
    if len(good_matches) > 4:
        # Extract location of good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        # Compute homography
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
        return H, keypoint_coordinates
    else:
        print("Not enough matches found to compute homography.")
        return None, keypoint_coordinates


def compute_direction(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect features to track
    features_to_track = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # Calculate optical flow (i.e., track feature points)
    points2, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, features_to_track, None)

    # Filter only points with high confidence
    good_new = points2[st==1]
    good_old = features_to_track[st==1]

    # Calculate movement
    movement_directions = good_new - good_old

    avg_direction = np.median(movement_directions, axis=0)
    avg_direction_normalized = avg_direction / np.linalg.norm(avg_direction)
    print(avg_direction_normalized)

    start_x, start_y = image1.shape[1] // 2, image1.shape[0]
    # Example: unit vector pointing right (you can change this as needed)
    u, v = -avg_direction_normalized[0] / 4, 1

    # Normalize the vector to make it a unit vector
    length = np.sqrt(u**2 + v**2)
    u, v = u / length, v / length

    # Plot the image
    plt.clf()
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    plt.imshow(image1)
    
    # # Plot the vector
    plt.quiver(start_x, start_y, u, v, scale=5, color='red')
    # # plt.show()
    # plt.pause(0.5)

    # Display the frame
    plt.pause(0.1)



    # Example: Print movement of first feature
    print("Movement of first feature: ", movement_directions[0])


def draw_vector(frame, start, vector, color=(0, 255, 0), thickness=2):
    """
    Draw a vector (as an arrow) on an image.
    
    :param frame: The image on which to draw.
    :param start: Starting point of the vector (x, y).
    :param vector: The vector to draw.
    :param color: Arrow color.
    :param thickness: Line thickness of the arrow.
    """
    end = (int(start[0] + vector[0] * 50), int(start[1] + vector[1] * 50))  # Scale vector for visibility
    cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.3)

def sort_images_numerically(file_list):
    # Function to extract the number from the filename
    def extract_number(filename):
        # Use a regular expression to find the first sequence of digits in the filename
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    # Sort the list using the extract_number function as the key
    return sorted(file_list, key=extract_number)



if __name__ == '__main__':

    parse_video = False
    output_folder = 'mountain_frames'        # Folder to save frames

    if parse_video:
        # Example usage
        video_path = 'mountain_drive.mp4'  # Path to your video file
        split_video_into_frames(video_path, output_folder)

    paths = get_image_paths(output_folder)
    file_list = [f for f in paths if f.startswith(f'{output_folder}/frame_') and f.endswith('.jpg')]

    # sorted files
    sorted_files = sort_images_numerically(file_list)

    for i in range(len(sorted_files)-1):

        image1_path = sorted_files[i]
        image2_path = sorted_files[i + 1]

        # H, coords = compute_homography(image1_path, image2_path)

        compute_direction(image1_path, image2_path)


        # print("Homography Matrix:")
        # print(H)


