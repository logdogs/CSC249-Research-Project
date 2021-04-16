import math
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import PIL.Image as im
import subprocess
import urllib

image = imageio.imread("test_character.png")
# It looks like first we have to take the input image (sigma) and do two things:
#   Shrink it
#   Convert it to either black and white or gray (idk if it requires bw or we can pass with grayscale)
def grayscale(image):
    return np.dot(image[:,:], [1/3,1/3,1/3])

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

def get_threshold(gray,avg_darkness,darkest_val):
    std_darkness = 0.0
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            std_darkness += (gray[i,j] - avg_darkness) ** 2
    std_darkness /= (gray.shape[0] + gray.shape[1])
    std_darkness = math.sqrt(std_darkness)
    threshold = (darkest_val - avg_darkness) / std_darkness
    threshold *= 1000
    return threshold

def create_binary_image(gray,threshold):
    image_prime = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i,j] > threshold:
                image_prime[i,j] = 0
            else:
                image_prime[i,j] = gray[i,j]
    return image_prime

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

def convert_for_CNN(image_name):
    image = imageio.imread(image_name)
    gray = grayscale(image)
    plt.figure()
    plt.imshow(gray)
    plt.show()
    avg_d,darkest = get_avg_and_darkest(gray)
    binary_image = create_binary_image(gray,get_threshold(gray,avg_d,darkest))
    cropped = bound_image(binary_image)
    resized = cv2.resize(cropped,(96,96),interpolation=cv2.INTER_CUBIC)
    return resized

def save(resized,name):
    # Assuming you added the file extension to the name
    cv2.imwrite(name,resized)

def run_CNN(input_name,save_name):
    # Using the notation of our formalization, input_image == \Sigma
    to_save = convert_for_CNN(input_name)
    save(to_save,save_name)
    subprocess.call("model_test.py " + save_name, shell=True)

run_CNN("test_character2.png", "test_resized2.png")
# plt.figure()
# plt.imshow(resized,cmap='gray')
# plt.show()
# cv2.imwrite("test_resized.png", resized)
