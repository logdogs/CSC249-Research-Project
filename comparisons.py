import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.metrics import structural_similarity as ss
import radical_segmentation as rs

# First, we need to get both of the images such that all character pixels are the same value, namely 1
def correct_pixel_vals(character):
    corrected = np.zeros_like(character)
    for i in range(character.shape[0]):
        for j in range(character.shape[1]):
            if character[i,j] != 0:
                corrected[i,j] = 1
    return corrected


# For data on the results of this:
#   https://docs.google.com/spreadsheets/d/1C2F6ATsXjz7HA5sOvN5NuVwXconegeas81zqWVPuz3Y/edit?usp=sharing
def structural_similarity(character, comparison):
    char_arr = np.asarray(character)
    comp_arr = np.asarray(comparison)
    corrected = correct_pixel_vals(char_arr)
    # Compare both the corrected image and the original (converted original, that is) to comp_arr
    
    og_comp = img_as_float(char_arr)
    rows,cols = og_comp.shape
    ssim1 = ss(comp_arr, og_comp)
    ssim2 = SSIM(comp_arr, og_comp)
    print(ssim1)
    print(ssim2)
    
    # plt.figure()
    # plt.imshow(comp_arr,cmap='gray')
    # plt.figure()
    # plt.imshow(og_comp,cmap='gray')
    # plt.show()


# So probably remove the luminocity and contrast calculations from the formulation of SSIM
#   If possible also add in (via multiplication as to fit the formulation), component for characters
#   (idk, maybe like something that looks at edges, stright lines, etc.)

# This is a straightforward definition from the formalization on:
#   https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
#   Critically note that x is an image
def luminance(x):
    mu_x = 0.0
    N = 96 * 96
    sigma = 0.0
    for i in range(96):
        for j in range(96):
            sigma += x[i,j]
    mu_x = (1/N) * sigma
    return mu_x
def contrast(x):
    sigma_x = 0.0
    N = 96 * 96
    sigma = 0.0
    mu_x = luminance(x)
    for i in range(96):
        for j in range(96):
            sigma += (x[i,j] - mu_x) ** 2
    sigma_x = ((1/(N-1)) * sigma) ** (1/2)
    return sigma_x
# How dare you question my function naming conventions...
def var(x,y):
    N = 96 * 96
    s = 0.0
    mu_x = luminance(x)
    mu_y = luminance(y)
    for i in range(96):
        for j in range(96):
            s += (x[i,j] - mu_x) * (y[i,j] - mu_y)
    s *= 1/(N-1)
    return s
def structure(x):
    mu_x = luminance(x)
    sigma_x = contrast(x)
    res_vect = np.copy(x)
    for i in range(96):
        for j in range(96):
            res_vect[i,j] = (x[i,j] - mu_x) / sigma_x
    return res_vect
# Now the comparison functions
def l(x,y):
    mu_x = luminance(x)
    mu_y = luminance(y)
    # Using the web's example of C_1 and C_2
    c1 = (0.01) ** 2
    return (2 * mu_x * mu_y + c1)/((mu_x ** 2) + (mu_y ** 2) + c1)
def c(x,y):
    sigma_x = contrast(x)
    sigma_y = contrast(y)
    c2 = (0.03) ** 2
    return (2 * sigma_x * sigma_y + c2)/((sigma_x ** 2) + (sigma_y ** 2) + c2)
def s(x,y):
    sigma_xy = var(x,y)
    sigma_x = contrast(x)
    sigma_y = contrast(y)

    c3 = ((0.03) ** 2) /2
    return (sigma_xy + c3)/(sigma_x * sigma_y + c3)
# The location component from my formalization (g for \Gamma)
# def g(x,y):
#     c_1 = []
#     c_2 = []
#     for i in range(96):
#         for j in range(96):
#             if x[i,j] != 0:
#                 c_1.append([i,j])
#             if y[i,j] != 0:
#                 c_2.append([i,j])
#     alpha_1 = 1 / len(c_1)
#     alpha_2 = 1 / len(c_2)
#     # As per the convention in the file, 's' is short for a sum
#     s_1 = 0.0
#     s_2 = 0.0
#     for pixel in c_1:
#         s_1 += sum(pixel)
#     for pixel in c_2:
#         s_2 += sum(pixel)
#     s_1 *= alpha_1
#     s_2 *= alpha_2
#     return abs(s_1 - s_2)

# SHOULD ALWAYS RETAIN THE SAME PASSING CONVENTION TO g(x,y):
#    x --> sigma
#    y --> sigma'
# I COULD FIX THIS AND SEMI-ENFORCE IT BY CHANGING THE VARIABLE NAMES BUT EH
def g(x,y):
    c_1 = []
    c_2 = []
    for i in range(96):
        for j in range(96):
            if x[i,j] != 0:
                c_1.append([i,j])
            if y[i,j] != 0:
                c_2.append([i,j])
    # Sum all location values
    #   *A location value is simple the sum of the x and y coordinates of a pixel
    # Return the quotient of s_1 and s_2 (following the naming convention of the formalization), denoted q
    #   Thus we have a very concrete idea of what the number means:
    #   0 \leq q \leq 1, where 1 is perfect, 0 and q \approx 0 is terrible
    s_1 = 0.0
    s_2 = 0.0
    for pixel in c_1:
        s_1 += sum(pixel)
    for pixel in c_2:
        s_2 += sum(pixel)
    # These normalizations make the values group much tighter for good, okay, and bad
    # Maybe useful, but I think this makes thresholds harder
    # alpha_1 = 1 / len(c_1)
    # alpha_2 = 1 / len(c_2)
    # s_1 *= alpha_1
    # s_2 *= alpha_2
    return min(s_1,s_2)/max(s_1,s_2)
# Proportion of the frame taken up by the character
def proportion_taken(x):
    total_size = 32 * 32
    to_ret = []
    base_arr = [[0, 0], [0, 32], [0, 64],
                [32, 0], [32, 32], [32, 64],
                [64, 0], [64, 32], [64, 64]]
    for b in range(len(base_arr)):
        for i in range(32):
            poz = 0
            for j in range(32):
                if x[i+base_arr[b][0],j+base_arr[b][1]] != 0:
                    poz += 1
            
        to_ret.append(poz / (32 * 32))
            # to_ret.append(poz)

    return to_ret
def p(x,y):
    prop_x = proportion_taken(x)
    prop_y = proportion_taken(y)

    overall_x = 0.0
    overall_y = 0.0
    for i in range(len(prop_x)):
        overall_x += (1/9) * prop_x[i]
        overall_y += (1/9) * prop_y[i]
    return min(overall_x,overall_y) / max(overall_x,overall_y)

    # metric = 0.0
    # comp_vect = []
    # for i in range(len(prop_x)):
    #     if prop_x[i] == 0 and prop_y[i] == 0:
    #         comp_vect.append(1)
    #     elif prop_x[i] == 0 and prop_y[i] != 0:
    #         comp_vect.append(0)
    #     elif prop_x[i] != 0 and prop_y[i] == 0:
    #         comp_vect.append(0)
    #     else:
    #         comp_vect.append(min(prop_x[i],prop_y[i])/max(prop_x[i],prop_y[i]))
    # print(comp_vect)
    # print(len(comp_vect))
    # return comp_vect
# The moment of truth (not bad! Just... different...)
def SSIM(x,y):
    # I'm not doing the simplified form for a very definitive reason
    # -->This allows us more ability to manipulate each of them in terms of which we think is more
    #       important/relevant to our comparison. I'm thinking about downplaying l and c
    # We can very easily just algebraically manipulate this to have way better (by (big) constant factors)
    #   run times. Right now ain't great, but not tragically bad
    # For the sake of run time (tho it ain't much better), I left out the exponents 
    file = open('cnn_output_character.txt', 'r', encoding='utf8')
    character = file.read()
    # return (l(x,y) ** (1/2)) * (c(x,y) ** (1/2)) * s(x,y) * g(x,y)
    return s(x,y) * g(x,y)
def overlay(x,y):
    # For a given pixel p, if:
    #   p \in x ^ p \notin y --> p will display as blue
    #   p \in y ^ p \notin x --> p will display as red
    #   p \in x ^ p \in y --> p will display as white
    overlay = np.zeros((96,96,3))
    for i in range(96):
        for j in range(96):
            if x[i,j] != 0 and y[i,j] != 0:
                overlay[i,j,:] = 255
            elif x[i,j] == 0 and y[i,j] != 0:
                overlay[i,j,0] = 255
            elif x[i,j] != 0 and y[i,j] == 0:
                overlay[i,j,1] = 255
    return overlay
def compare_radical_sizes(x,y):
    # compare the shape of x and y to see how proportional they've written a given radical
    # Take x to be from sigma, take y to be from sigma'
    length_proportion = min(x.shape[0],y.shape[0]) / max(x.shape[0],y.shape[0])
    height_proportion = min(x.shape[1],y.shape[1]) / max(x.shape[1],y.shape[1])

# Get the weight for a component of a multi-component character
# In the formalization the names are exactly the same 
def get_alpha(Gamma_before, delta_before):
    # 'card' is short for cardinality
    Gamma = np.asarray(Gamma_before)
    delta = np.asarray(delta_before)
    Gamma_card = 0
    delta_card = 0
    for i in range(96):
        for j in range(96):
            if Gamma[i,j] != 0:
                Gamma_card += 1
            if delta[i,j] != 0:
                delta_card += 1
    return delta_card / Gamma_card

# Compare a character and it's printed version: sigma : sigma'
# Two cases for sigma/sigma':
#   1) Base case: 1 Component, no segmentation
#   2) Inductive case:  > 1 component, so segmentation
def compare(sigma,sigma_prime):
    sigma_seg = rs.segment(sigma)
    sigma_prime_seg = rs.segment(sigma_prime)
    
    sigma_seg[0].show()

    if len(sigma_seg) > 1: # case 2
        component_pairs = []
        for i in range(len(sigma_seg)):
            component_pairs.append((sigma_seg[i],sigma_prime_seg[i]))
        # component_comp_vals = collections.defaultdict(tuple)
        component_comp_vals = []
        # component_comp_vals = collections.defaultdict(tuple)
        for pair in component_pairs:
            # Recall that pair[0] is always simga, and pair[1] is always sigma'
            # We will have component_comp_vals have list always of the form [p,s,g]
            # component_comp_vals[pair] = [p(np.asarray(list(pair)[0]),np.asarray(list(pair)[1])),s(np.asarray(list(pair)[0]),np.asarray(list(pair)[1])),g(np.asarray(list(pair)[0]),np.asarray(list(pair)[1]))]
            component_comp_vals.append([list(pair)[0], list(pair)[1], p(np.asarray(list(pair)[0]),np.asarray(list(pair)[1])),s(np.asarray(list(pair)[0]),np.asarray(list(pair)[1])),g(np.asarray(list(pair)[0]),np.asarray(list(pair)[1]))])

        overall_p_val = 0.0
        overall_s_val = 0.0
        overall_g_val = 0.0
        #for keys, vals in component_comp_vals.items():
        for vals in component_comp_vals:
            alpha = get_alpha(sigma, vals[0])
            overall_p_val += vals[2] * alpha
            overall_s_val += vals[3] * alpha
            overall_g_val += vals[4] * alpha
    else: # Case 1
        overall_p_val = p(sigma,sigma_prime)
        overall_s_val = s(sigma,sigma_prime)
        overall_g_val = g(sigma,sigma_prime)
    print("p -", overall_p_val)
    print("s -", overall_s_val)
    print("g -", overall_g_val)
    return overall_p_val, overall_s_val, overall_g_val
        