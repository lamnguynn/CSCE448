# Import required libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


# Read source and mask (if exists) for a given id
def Read(id, path = ""):
    source = plt.imread(path + "image_" + id + ".jpg")
    maskPath = path + "mask_" + id + ".jpg"
    
    if os.path.isfile(maskPath):
        mask = plt.imread(maskPath)
        assert(mask.shape == source.shape), 'size of mask and image does not match'
        mask = (mask > 128)[:, :, 0].astype(int)
    else:
        mask = np.zeros_like(source)[:, :, 0].astype(int)

    return source, mask


def compute_energy_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    energy = np.abs(dx) + np.abs(dy)

    return energy


def compute_optimal_seam(E):
    # Make the scoring matrix M
    M = E.copy()

    # Set the values of every entry in the matrix except for the first row
    for i in range(1, M.shape[0]):
        for j in range(M.shape[1]):
            # Need to compensate for bounds
            if j == 0:
                M[i][j] = E[i][j] + min(M[i - 1][j], M[i - 1][j + 1])
            elif j == M.shape[1] - 1:
                M[i][j] = E[i][j] + min(M[i - 1][j - 1], M[i - 1][j])
            else:
                M[i][j] = E[i][j] + min(M[i - 1][j - 1], M[i - 1][j], M[i - 1][j + 1])
    return M

def find_min_value(M):
    to_list = list(M)
    return to_list.index(min(to_list))

def compute_seam_path(M):
    h, w = M.shape
    min_val_bot_rpw = find_min_value(M[h - 1, :])
    seam = [min_val_bot_rpw]
    i = 0

    for y in range(h - 1, 0, -1):
        x = seam[i]

        if (x == 0):
            min_val = find_min_value(M[y - 1, x : x + 2])
            seam.append(x + min_val)
        elif(x == w - 1):
            min_val = find_min_value(M[y - 1, x - 1: x + 1])
            seam.append(x + min_val - 1)
        else:
            min_val = find_min_value(M[y - 1, x - 1: x + 2])
            seam.append(x + min_val - 1)
        i += 1

    return seam

def seam_removal(img, seam):
    h, w, _ = img.shape
    output = cv2.resize(img, (w - 1, h), interpolation=cv2.INTER_LINEAR)
    col = h - 1

    for i in range(img.shape[0]):
        row = seam[i]
        output[col, : row] = img[col, : row]
        output[col, row:] = img[col, row + 1:]
        col = col - 1

    return output

def scale_column(input, scale_c):
    target_width = int(input.shape[1] * scale_c)

    for i in range(input.shape[1] - target_width):
        E = compute_energy_matrix(input)
        M = compute_optimal_seam(E)
        seam = compute_seam_path(M)
        input = seam_removal(input, seam)

    return input

def scale_row(input, scale_r):
    input = np.rot90(input, 1)
    input = scale_column(input, scale_r)

    return np.rot90(input, 3)

def SeamCarve(input, widthFac, heightFac, mask):

    # Main seam carving function. This is done in three main parts: 1)
    # computing the energy function, 2) finding optimal seam, and 3) removing
    # the seam. The three parts are repeated until the desired size is reached.

    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'

    if heightFac != 1:
        input = scale_row(input, heightFac)
    else:
        input = scale_column(input, widthFac)

    return input, (input.shape[1], input.shape[0])

# Setting up the input output paths
inputDir = '../Images/'
outputDir = '../Results/'

widthFac = 1; # To reduce the width, set this parameter to a value less than 1
heightFac = 0.5;  # To reduce the height, set this parameter to a value less than 1
N = 4 # number of images

for index in range(N, N + 1):

    input, mask = Read(str(index).zfill(2), inputDir)

    # Performing seam carving. This is the part that you have to implement.
    output, size = SeamCarve(input, widthFac, heightFac, mask)

    # Writing the result
    plt.imsave("{}/result_{}_{}x{}.jpg".format(outputDir, 
                                            str(index).zfill(2), 
                                            str(size[0]).zfill(2), 
                                            str(size[1]).zfill(2)), output)