import decomp_lookup as dl
import numpy as np
from PIL import Image
import imageio
import os
import math


#Finds longest continuous subsequence in array
def longest_run(arr):
    longest_run = []
    current_run = []

    for x in reversed(arr):
        if len(current_run) == 0:
            current_run.append(x)
        else:
            if x == current_run[len(current_run) - 1] - 1:
                current_run.append(x)
            else:
                if len(current_run) > len(longest_run):
                    longest_run = current_run
                    current_run = []
                else:
                    current_run = []
    
    if len(longest_run) == 0:
        longest_run = current_run
    
    return longest_run


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

def get_lowest_point(data):
    for i in reversed(range(data.shape[0])):
        for j in range(data.shape[1]):
            if data[i,j] != 0:
                return i

def get_highest_point(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] != 0:
                return i

def get_left_point(data):
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            if data[i,j] != 0:
                return j

def get_right_point(data):
    for j in reversed(range(data.shape[1])):
        for i in range(data.shape[0]):
            if data[i,j] != 0:
                return j


# Finds best column to segment top-down oriented radical
def down(image, orientation):
    row_vals = {}
    data = np.asarray(image)
    height = data.shape[0]
    width = data.shape[1]

    highest = get_highest_point(data)
    lowest = get_lowest_point(data)
    adjusted_height = lowest - highest
    lower_bound = int(.15 * adjusted_height + highest) 
    upper_bound = int(.85 * adjusted_height + highest) 
    if orientation == "b":
        upper_bound = int(.5 * adjusted_height + highest)
    if orientation == "t":
        lower_bound = int(.5 * adjusted_height + highest)
    for x in range(height):
        count = 0
        if x > lower_bound and x < upper_bound:
            for y in range(width):
                if data[x][y] > 0:
                    count += 1
            row_vals[x] = count
    temp = min(row_vals.values())
    res = [key for key in row_vals if row_vals[key] == temp]
    longest = longest_run(res)
    average = sum(longest) / len(longest)
    
    top = (0, 0, width, average)

    bottom = (0, average, width, height)

    return top, bottom
    
# Finds best column to segment left-right oriented radical
def across(image, orientation):
    col_vals = {}
    data = np.asarray(image)
    height = data.shape[0]
    width = data.shape[1]

    left_most = get_left_point(data)
    right_most = get_right_point(data)
    adjusted_width = right_most-left_most
    lower_bound = int(.2 * adjusted_width + left_most) 
    upper_bound = int(.8 * adjusted_width + left_most) 
    if orientation == "l":
        upper_bound = int(0.5 * adjusted_width + left_most(data))
    if orientation == "r":
        lower_bound = int(0.5 * adjusted_width + left_most)
    for x in range(width):
        count = 0
        if x > lower_bound and x < upper_bound:
            for y in range(height):
                if data[y][x] > 0:
                    count += 1
            col_vals[x] = count

    temp = min(col_vals.values())
    res = [key for key in col_vals if col_vals[key] == temp]

    longest = longest_run(res)

    average = sum(longest) / len(longest)

    left = (0, 0, average, height)
    right = (average, 0, width, height)

    return left, right

# Combines down and accross
def s_surround(image, ud_orientation, lr_orientation):
    top, bottom = down(image, ud_orientation)
    left, right = across(image, lr_orientation)
    
    l = 0
    r = 0
    u = 0
    d = 0

    if ud_orientation == "b":
        d = top[3]

    if ud_orientation == "t":
        u = top[3]

    if lr_orientation == "l":
        l = right[0]
        r = right[2]
    if lr_orientation == "r":
        l = right[2]
        r = right[0]

    dims = (l, u, r, d)
    return dims

# Combines down and accross
def w_contained(image):
    top_b, bottom_b = down(image, "b")
    top_u, bottom_u = down(image, "u")
    left_l, right_l = across(image, "l")
    left_r, right_r = across(image, "r")
    
    l = right_l[0]
    r = right_r[0]
    u = top_b[3]
    d = bottom_u[2]

    dims = (l, u, r, d)
    return dims

def remove(img, dims):
    img_arr = np.array(img)
    l = int(dims[0])
    r = int(dims[2])
    u = int(dims[1])
    d = int(dims[3])
    img_arr[u : d, l : r] = (0)
    
    # Creating an image out of the previously modified array
    img = Image.fromarray(img_arr)
    
    return img

def display(image, display_bool):
    if display_bool:
        image.show()

# Pain
# Simply returns images of either *the* isolated component, or both of them
def isolate(x, bounds_list):
    x = np.asarray(x)
    isolated_list = []
    isolated = np.zeros_like(x)
    if len(bounds_list) == 1:
        # Can be contained or surrounds
        bound = list(bounds_list[0])
        for i in range(int(math.ceil(bound[0])),int(math.ceil(bound[2]))):
            for j in range(int(math.ceil(bound[1])),int(math.ceil(bound[3]))):
                isolated[j,i] = x[j,i]
        isolated = Image.fromarray(isolated)
        isolated_list.append(isolated)
    elif len(bounds_list) == 2:
        # Can be across or it can be down
        # bound1 will get the component on the left, bound2 the component on the right?
        #   --> thus we return two images?
        bound1 = bounds_list[0]
        bound2 = bounds_list[1]
        res1 = np.zeros_like(x)
        res2 = np.zeros_like(x)
        for i in range(int(math.ceil(bound1[0])),int(math.ceil(bound1[2]))):
            for j in range(int(math.ceil(bound1[1])),int(math.ceil(bound1[3]))):
                res1[j,i] = x[j,i]
        for i in range(int(math.ceil(bound2[0])),int(math.ceil(bound2[2]))):
            for j in range(int(math.ceil(bound2[1])),int(math.ceil(bound2[3]))):
                res2[j,i] = x[j,i]
        res1 = Image.fromarray(res1)
        res2 = Image.fromarray(res2)
        isolated_list.append(res1)
        isolated_list.append(res2)
    else:
        print("uhoh")
        os.exit()
    return isolated_list

def run(image, character, display_bool, final_list):
    possibilities = "das"
    orientation = "lt"
    d = dl.decomp_dictionary()
    structure = d.get_composition_structure(character)
    if character != '𧰨' and structure[0] in possibilities:
        components = d.get_components(character)
        if len(components) > 1:
            if structure == "d": #down
                top_char = components[0]
                bottom_char = components[1]
                top, bottom = down(image, "n") #top/bottom dims
                isolated_components = isolate(image,[top,bottom])
                top = isolated_components[0]
                bottom = isolated_components[1]

                display(top, display_bool)
                display(bottom, display_bool)
                t = run(top, top_char, display_bool, final_list)
                b = run(bottom, bottom_char, display_bool, final_list)
            
            if structure == "a": #across
                left, right = across(image, "n") #left/right dims
                isolated_components = isolate(image,[left,right])
                left = isolated_components[0]
                right = isolated_components[1]
                display(left, display_bool)
                display(right, display_bool)
                l = run(left, components[0], display_bool, final_list)
                r = run(right, components[1], display_bool, final_list)

            if structure[0] == "s": #surround
                if structure[1] is not None:
                    if structure[2] is not None:
                        s = s_surround(image, structure[1], structure[2]) #inner dims
                        isolated = isolate(image,[s])
                        inner = isolated[0]
                        display(inner, display_bool)
                        outer = remove(image, s)
                        display(outer, display_bool)
                        o = run(outer, components[0], display_bool, final_list)
                        s = run(s, components[1], display_bool, final_list)
                else: #equivalent to withtin
                    w = w_contained(image)
                    isolated_components = isolate(image,[w])
                    inner = isolated_components[0]
                    display(inner, display_bool)
                    outer = remove(image, w)
                    display(outer, display_bool)
                    i = run(inner, components[0], display_bool, final_list)
                    o = run(outer, components[1], display_bool, final_list)

            if structure[0] == "w": #within
                w = w_contained(image) #inner dims
                isolated_components = isolate(image,[w])
                inner = isolated_components[0]
                display(inner, display_bool)
                outer = remove(image, w)
                display(outer, display_bool)
                i = run(inner, components[0], display_bool, final_list)
                o = run(outer, components[1], display_bool, final_list)
        else:
            if type(image) is not tuple:
                final_list.append(image)         
    else:
        if type(image) is not tuple:
            final_list.append(image)

def segment(image):
    file = open('cnn_output_character.txt', 'r', encoding='utf8')
    character = file.read()
    final_list = []
    run(image, character, False, final_list)
    return final_list
