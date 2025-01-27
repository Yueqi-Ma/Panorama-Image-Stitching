import cv2
import numpy as np
import matplotlib.pyplot as plt


# ================================================================== #
#                     Feature extractor function
# ================================================================== #
def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
    assert method is not None, "You need to define a feature detection method."

    # Detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    # Additional methods can be added here
    elif method == 'orb':
        descriptor = cv2.ORB_create()


    # Get keypoints and descriptors
    kps, features = descriptor.detectAndCompute(image, None)
    return (kps, features)


# ================================================================== #
#                     Matcher Function
# ================================================================== #
def createMatcher(method, crossCheck):
    "Create and return a Matcher Object"
    if method == 'sift':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    # Additional methods can be added here
    elif method == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)

    return bf


# ================================================================== #
#                     Homography Calculation Function
# ================================================================== #
def getHomography(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4):
    # Match descriptors.
    matcher = createMatcher('sift', crossCheck=False)
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    # Filter matches using the Lowe's ratio test
    good = []
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            good.append(m)
    if len(good) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good])
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (good, H, status)
    else:
        return None


# ================================================================== #
#                     Image Stitching Function
# ================================================================== #
def stitchImages(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and extract
    kpsA, featuresA = detectAndDescribe(gray1, method='orb')
    kpsB, featuresB = detectAndDescribe(gray2, method='orb')

    # display keypoint
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), constrained_layout=False)
    ax1.imshow(cv2.drawKeypoints(gray1, kpsA, None, color=(0, 255, 0)))
    ax1.set_xlabel("(a) Key Points", fontsize=14)
    ax2.imshow(cv2.drawKeypoints(gray2, kpsB, None, color=(0, 255, 0)))
    ax2.set_xlabel("(b) Key Points", fontsize=14)
    plt.show()


    # Match features between the two images
    M = getHomography(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4)
    if M is None:
        print("Error! - Homography could not be computed")
        return None
    (matches, H, status) = M

    # disply matches
    img_matches = cv2.drawMatches(img1, kpsA, img2, kpsB, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title("Matches between Images")
    plt.show()


    # Apply panorama correction
    width = img1.shape[1] + img2.shape[1]
    height = max(img1.shape[0], img2.shape[0])
    result = cv2.warpPerspective(img1, H, (width, height))


    # 调试打印：输出变换后的图像尺寸
    print("Transformed img1 size:", result.shape)

    # 计算重叠区域的宽度
    overlap_start = 0
    overlap_end = img2.shape[1]

    # # 使用高斯函数改进混合权重
    # x = np.linspace(-3, 3, overlap_end - overlap_start)
    # alpha = np.exp(-x**2)
    sigma = 1  # 可以调整这个值
    x = np.linspace(-3 * sigma, 3 * sigma, overlap_end - overlap_start)
    alpha = np.exp(-x ** 2 / (2 * sigma ** 2))

    # 在重叠区域应用混合
    for row in range(img2.shape[0]):
        for col in range(overlap_start, overlap_end):
            if np.any(result[row, col] != 0):
                beta = 1 - alpha[col - overlap_start]
                result[row, col] = alpha[col - overlap_start] * img2[row, col] + beta * result[row, col]

    # 复制剩余的非重叠部分
    result[0:img2.shape[0], 0:img2.shape[1]] = np.maximum(img2, result[0:img2.shape[0], 0:img2.shape[1]])

    # Crop the result
    result = crop(result)

    return result


# ================================================================== #
#                     Image Cropping Function
# ================================================================== #
def crop(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Find contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get rectangle bounding contour
    x, y, w, h = cv2.boundingRect(contours[0])
    # Crop
    crop = image[y:y + h, x:x + w]
    return crop



# # Load the images
# img1 = cv2.imread('DSC_0171.jpg')
# img2 = cv2.imread('DSC_0172.jpg')

# Load the images
img1 = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')

# Stitch the images
stitched_image = stitchImages(img2, img1)

# Check if the stitching process is successful
if stitched_image is not None:
    # Save the stitched image
    stitched_image_path = '/mnt/data/stitched_image.jpg'
    cv2.imwrite(stitched_image_path, stitched_image)

    # Convert to RGB for matplotlib display
    stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)

    # Display the result
    plt.imshow(stitched_image_rgb)
    plt.axis('on')  # Turn off axis numbers and ticks
    plt.title('Stitched Image')
    plt.show()

    # Provide the path for the saved image
    stitched_image_path
else:
    print("Stitching failed.")
    stitched_image_path = None
