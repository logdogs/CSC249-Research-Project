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

from skimage.morphology import skeletonize
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import numpy
"load image data"

"Convert gray images to binary images using Otsu's method"
from skimage.filters import threshold_otsu
#Otsu_Threshold = threshold_otsu(Img_Original)   
#BW_Original = Img_Original < Otsu_Threshold    # must set object region as 1, background region as 0 !

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned
 




def main():
    args = sys.argv
    assert len(args) == 2, "Should run as 'python primer.py <image_of_character>'"
    input_name = args[1]
    file_name = input_name.split('.')[0]
    file_name += "_resized.png"
    run_CNN(input_name,file_name)
    sigma = convert_for_CNN(input_name)
    sigma_prime = imageio.imread("res.png")


    #sigma_skel = sigma      # Gray image, rgb images need pre-conversion
    #sigma[sigma!=0] = 1
    #sigma_skel = skeletonize(sigma)
    sigma = cv2.GaussianBlur(sigma,(3,3),0)
    th, sigma = cv2.threshold(sigma, 1, 255, cv2.THRESH_BINARY_INV)
    #sigma_skel = numpy.invert(sigma.astype(np.int))
    #sigma_skel = numpy.invert(sigma)
    sig_Threshold = threshold_otsu(sigma)
    sigma_skel = sigma < sig_Threshold
    sigma_skel = zhangSuen(sigma_skel)

    #sigma_prime[sigma_prime!=0] = 1
    #sigma_prime_skel = skeletonize(sigma_prime)
    #sigma_prime_skel =  sigma_prime      # Gray image, rgb images need pre-conversion
    sigma_prime_skel = numpy.invert(sigma_prime.astype(np.int))
    sig_prime_Threshold = threshold_otsu(sigma_prime_skel)
    sigma_prime_skel = sigma_prime_skel < sig_prime_Threshold
    sigma_prime_skel = zhangSuen(sigma_prime_skel)
    
    "Apply the algorithm on images"
    #BW_Skeleton = zhangSuen(BW_Original)
    # BW_Skeleton = BW_Original

    "Display the results"
    fig, ax = plt.subplots(1, 2)
    ax1, ax2 = ax.ravel()
    ax1.imshow(sigma, cmap=plt.cm.gray)
    ax1.set_title('sigma')
    ax1.axis('off')
    ax2.imshow(sigma_skel, cmap=plt.cm.gray)
    ax2.set_title('Skeleton of the image')
    ax2.axis('off')
    plt.show()

    fig, ax = plt.subplots(1, 2)
    ax1, ax2 = ax.ravel()
    ax1.imshow(sigma_prime, cmap=plt.cm.gray)
    ax1.set_title('sigma_prime')
    ax1.axis('off')
    ax2.imshow(sigma_prime_skel, cmap=plt.cm.gray)
    ax2.set_title('Skeleton of the image')
    ax2.axis('off')
    plt.show()


    comparisons.structural_similarity(sigma_skel,sigma_prime_skel)
    # comparisons.ombb(sigma)
    # comparisons.ombb(sigma_prime)
    # plt.figure()
    # plt.imshow(sigma,cmap='gray')
    # plt.show()
    # os.remove(file_name)
    

main()