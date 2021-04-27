import numpy as np
from PIL import Image
import imageio
import cld_lookup
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.metrics import structural_similarity as ss
import decomp_lookup
import math
import scipy.spatial as sp

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
def g(x,y):
    c_1 = []
    c_2 = []
    for i in range(96):
        for j in range(96):
            if x[i,j] != 0:
                c_1.append([i,j])
            if y[i,j] != 0:
                c_2.append([i,j])
    alpha_1 = 1 / len(c_1)
    alpha_2 = 1 / len(c_2)
    # As per the convention in the file, 's' is short for a sum
    s_1 = 0.0
    s_2 = 0.0
    for pixel in c_1:
        s_1 += sum(pixel)
    for pixel in c_2:
        s_2 += sum(pixel)
    s_1 *= alpha_1
    s_2 *= alpha_2
    return abs(s_1 - s_2)
# Again, note that x denotes an image and will assume that only one character is in the image, if you want
#   it to just get a radical, just pass the trimmed image
def ombb(x_unpadded):
    # We must pad the image a bit so we can get an idea about the rotation
    x = np.zeros((176,176)) # 40 pixels of padding on top and on bottom
    for i in range(40,96+40):
        for j in range(40,96+40):
            i_temp = i - 40
            j_temp = j - 40
            x[i,j] = x_unpadded[i-40,j-40]
    
    character_points = []
    for i in range(96):
        for j in range(96):
            if x[i,j] != 0:
                character_points.append([i,j])
    ch = sp.ConvexHull(character_points)
    return minBoundingRect(ch.points)
def minBoundingRect(hull_points_2d):
    #print "Input convex hull points: "
    #print hull_points_2d

    # Compute edges (x2-x1,y2-y1)
    edges = np.zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]
    #print "Edges: \n", edges

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.zeros( (len(edges)) ) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = math.atan2( edges[i,1], edges[i,0] )
    #print "Edge angles: \n", edge_angles

    # Check for angles in 1st quadrant
    for i in range( len(edge_angles) ):
        edge_angles[i] = abs( edge_angles[i] % (math.pi/2) ) # want strictly positive answers
    #print "Edge angles in 1st Quadrant: \n", edge_angles

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)
    #print "Unique edge angles: \n", edge_angles

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, float("inf"), 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    # print( "Testing"), len(edge_angles), "possible rotations for bounding box... \n"
    for i in range( len(edge_angles) ):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = np.array([ [ math.cos(edge_angles[i]), math.cos(edge_angles[i]-(math.pi/2)) ], [ math.cos(edge_angles[i]+(math.pi/2)), math.cos(edge_angles[i]) ] ])
        #print "Rotation matrix for ", edge_angles[i], " is \n", R

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn
        #print "Rotated hull points are \n", rot_points

        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)
        #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width*height
        #print "Potential bounding box ", i, ":  width: ", width, " height: ", height, "  area: ", area 

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )
        # Bypass, return the last found rect
        #min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]   
    R = np.array([ [ math.cos(angle), math.cos(angle-(math.pi/2)) ], [ math.cos(angle+(math.pi/2)), math.cos(angle) ] ])
    #print "Projection matrix: \n", R

    # Project convex hull points onto rotated frame
    proj_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn
    #print "Project hull points are \n", proj_points

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]
    #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

    # Calculate center point and project onto rotated frame
    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    center_point = np.dot( [ center_x, center_y ], R )
    #print "Bounding box center point: \n", center_point

    # Calculate corner points and project onto rotated frame
    corner_points = np.zeros( (4,2) ) # empty 2 column array
    corner_points[0] = np.dot( [ max_x, min_y ], R )
    corner_points[1] = np.dot( [ min_x, min_y ], R )
    corner_points[2] = np.dot( [ min_x, max_y ], R )
    corner_points[3] = np.dot( [ max_x, max_y ], R )
    #print "Bounding box corner points: \n", corner_points

    #print "Angle of rotation: ", angle, "rad  ", angle * (180/math.pi), "deg"
    for point in corner_points:
        print (point)
    print("The rotation angle: " + str(angle))
    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points) # rot_angle, area, width, height, center_point, corner_points

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