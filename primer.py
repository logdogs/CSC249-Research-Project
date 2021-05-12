import math
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import PIL.Image as im
import subprocess
import sys
import os
import radical_segmentation as rs
import scipy.ndimage as nd

from numpy.core.arrayprint import ComplexFloatingFormat
import comparisons

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
    print(image_name)
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

# Source of this algorithm:
#   https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331
def skeletonize(image):
    ret,img = cv2.threshold(image,0,255,0)
    size = np.size(img)
    skel = np.zeros_like(img)
    element =  cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img,open)
        eroded = cv2.erode(img,element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

# Checks for a given pixel whether or no there is another character pixel in any of the 8 cells around it
def has_adjacent_character_pixel(image,x,y):
    if image[x-1,y] != 0 or image[x,y-1] != 0 or image[x-1,y-1] != 0 or image[x+1,y] != 0 or image[x,y+1] != 0 or image[x+1,y+1] != 0:
        return True
    else:
        return False

# Removes "noise" which is just a remnant of thickness (we don't want this because we want our comparisons to
#   be thickness-independent. That is, a character can still be well written even if the strokes are thick)
def clean_image(image):
    cleaned = np.copy(image)
    for i in range(1,95):
        for j in range(1,95):
            if not has_adjacent_character_pixel(image,i,j):
                cleaned[i,j] = 0
    return cleaned

# Function for test_all_images.py
#   *** IMPORTANT ***
#   For this to work, you must comment out the call to main() at the very end of this file.
def run(file):
    file_name = file.split('.')[0]
    file_name += "_resized.png"
    run_CNN(file,file_name)
    sigma = convert_for_CNN(file)
    sigma_prime = imageio.imread("res.png")
    sigma_skel = skeletonize(sigma)
    sigma_prime_skel = skeletonize(sigma_prime)
    
    p, s, g = comparisons.compare(sigma_skel,sigma_prime_skel)
    file = open('cnn_output_character.txt', 'r', encoding='utf8')
    character = file.read()

    os.remove(file_name)

    return p, s, g, character

def main():
    args = sys.argv
    assert len(args) == 2, "Should run as 'python primer.py <image_of_character>'"
    input_name = args[1]
    file_name = input_name.split('.')[0]
    file_name += "_resized.png"
    run_CNN(input_name,file_name)
    sigma = convert_for_CNN(input_name)
    sigma_prime = imageio.imread("res.png")

    sigma_skel = clean_image(skeletonize(sigma))
    sigma_prime_skel = clean_image(skeletonize(sigma_prime))

    comparisons.compare(sigma_skel,sigma_prime_skel)
    
    fig,axs = plt.subplots(1,3)
    axs1,axs2,axs3 = axs.ravel()
    axs1.imshow(sigma_skel,cmap='gray')
    axs1.set_title('sigma_skel')
    axs1.axis('off')
    axs2.imshow(sigma_prime_skel,cmap='gray')
    axs2.set_title('sigma_prime_skel')
    axs2.axis('off')
    axs3.imshow(comparisons.overlay(sigma_skel,sigma_prime_skel))
    axs3.set_title('overlay')
    axs3.axis('off')
    plt.savefig("for_presentation.png")
    plt.show()
    sigma_seg = rs.segment(im.fromarray(sigma_skel))
    for i in sigma_seg:
        print(i)
    sigma_prime_seg = rs.segment(im.fromarray(sigma_prime_skel))
    for i in sigma_prime_seg:
        print(i)
    # Cleanup the intermediate file created for sigma_prime
    os.remove(file_name)
#main()