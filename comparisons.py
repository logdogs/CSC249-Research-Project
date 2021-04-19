import numpy as np
from PIL import Image
import imageio
import cld_lookup
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ss
from skimage.metrics import mean_squared_error

# First, we need to get both of the images such that all character pixels are the same value, namely 1
def correct_pixel_vals(character):
    corrected = np.zeros_like(character)
    for i in range(character.shape[0]):
        for j in range(character.shape[1]):
            if character[i,j] != 0:
                corrected[i,j] = 1
    return corrected

def structural_similarity(character, comparison):
    char_arr = np.asarray(character)
    comp_arr = np.asarray(comparison)
    corrected = correct_pixel_vals(char_arr)
    # Compare both the corrected image and the original (converted original, that is) to comp_arr
    
    og_comp = img_as_float(char_arr)
    rows,cols = og_comp.shape
    ssim = ss(comp_arr, og_comp)
    
    print(ssim)
# da_bad --> .46
# da_okay --> .558
# da_good --> .529
# da_perfect --> 