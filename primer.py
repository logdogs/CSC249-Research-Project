import math
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import PIL.Image as im
import subprocess
import decomp_lookup
import sys
import comparisons
import os

# Function to remove the shadows from an image so that our method for finding the character will work
def remove_shadows(image):
    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    # result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm

# Simple grayscale calculation function
def grayscale(image):
    return np.dot(image[:,:], [1/3,1/3,1/3])

# Pretty straightforward, calculates the average darkness value (brightness) as well as the darkest value.
# I combined them into one function because it's just more efficient to calculate them both at the same
#   time, rather than add another \Theta(n^2) amount of run time, where n is the size of the image.
def get_avg_and_darkest(gray):
    avg_darkness = 0.0
    darkest_val = 0.0
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i,j] > darkest_val:
                darkest_val = gray[i,j]
            avg_darkness += gray[i,j]
    avg_darkness = avg_darkness / (gray.shape[0] * gray.shape[1])
    return avg_darkness,darkest_val

# This is not a very sophisticated method because I am not a clever man. It basically just calculates the
#   threshold as a normalized standardized score (C * Z, where C=1000 and Z=(X - \mu) / \sigma)
#   But why did I use 1000 and why Z? Well, because it worked when I tried it... again, not a clever man.
#   (now in my defence, 1000 because it got a really good value from Z which I could tell had potential)
def get_threshold(gray,avg_darkness,darkest_val):
    std_darkness = 0.0
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            std_darkness += (gray[i,j] - avg_darkness) ** 2
    std_darkness /= (gray.shape[0] * gray.shape[1])
    std_darkness = math.sqrt(std_darkness)
    threshold = (darkest_val - avg_darkness) + std_darkness
    # threshold *= 1000
    return threshold

# Okay, not exactly binary anymore, but originally it was. The CNN interestingly handled the binary images
#   very poorly (it was obvious from the confidence states (0 <= x_max <= 20) it had no idea which it was).
#   I took a gander at the data we were given with the CNN and the values of pixels in it, and found out
#   that that (\forall_{p \in I})[(p \in character) \iff (0 < p < 1)]. So I just changed it such that if
#   the pixel passes the threshold, it retains its value. Otherwise, it's 0 (black). 
def create_binary_image(gray,threshold):
    image_prime = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i,j] > threshold:
                image_prime[i,j] = 0
            else:
                image_prime[i,j] = gray[i,j]
    return image_prime

# Finds the exact bounding box for the character in the image. Do note that it expects only one character
#   to be in the image.
def bound_image(bin_image):
    max_x = 0
    max_y = 0
    min_x = 10000000
    min_y = 10000000
    for i in range(bin_image.shape[0]):
        for j in range(bin_image.shape[1]):
            if bin_image[i,j] > 0:
                if i > max_x:
                    max_x = i
                if i < min_x:
                    min_x = i
                if j > max_y:
                    max_y = j
                if j < min_y:
                    min_y = j
    image_prime = bin_image[min_x:max_x, min_y:max_y]
    return image_prime

# Will give you the resized and properly formatted version of the original image based on the filename
def convert_for_CNN(image_name):
    image = imageio.imread(image_name)
    gray = grayscale(image)
    avg_d,darkest = get_avg_and_darkest(gray)
    binary_image = create_binary_image(gray,get_threshold(gray,avg_d,darkest))
    cropped = bound_image(binary_image)
    resized = cv2.resize(cropped,(96,96),interpolation=cv2.INTER_CUBIC)
    return resized

# Quick save function that probably wasn't necessary
def save(resized,name):
    # Assuming you added the file extension to the name
    cv2.imwrite(name,resized)

# Call the CNN with our \Sigma
def run_CNN(input_name,save_name):
    # Using the notation of our formalization, input_image == \Sigma
    to_save = convert_for_CNN(input_name)
    save(to_save,save_name)
    subprocess.call("python model_test.py " + save_name, shell=True)


def main():
    args = sys.argv
    assert len(args) == 2, "Should run as 'python primer.py <image_of_character>'"
    input_name = args[1]
    file_name = input_name.split('.')[0]
    file_name += "_resized.png"
    run_CNN(input_name,file_name)
    sigma = convert_for_CNN(input_name)
    sigma_prime = imageio.imread("res.png")
    comparisons.structural_similarity(sigma,sigma_prime)
    # comparisons.ombb(sigma)
    # comparisons.ombb(sigma_prime)
    # plt.figure()
    # plt.imshow(sigma,cmap='gray')
    # plt.show()
    # os.remove(file_name)
main()