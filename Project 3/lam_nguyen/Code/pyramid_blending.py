# Acknowledgements: https://becominghuman.ai/image-blending-using-laplacian-pyramids-2f8e9982077f

# Import required libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read source, target and mask for a given id

# Evan Rachel Wood (right) and c (left)
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target

def ConvertToPyramids(image, levels):
    gauss = [image]
    lap = []

    #Find the Gaussian Pyramid
    G = image.copy()
    gauss = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gauss.append(G)

    #Find the Laplacian Pyramid
    lap.append(gauss[-1])
    for i in range(levels, 0, -1):
        G = cv2.pyrUp(gauss[i], dstsize = (gauss[i - 1].shape[1], gauss[i - 1].shape[0]))
        lap.append(gauss[i - 1] - G)

    return gauss, lap

# Pyramid Blend
def PyramidBlend(source, mask, target):
    levels = 9

    # Compute Laplacian pyramids LS and LT from the source and target images.
    # Compute a Gaussian pyramid GM from the mask.
    _, LS = ConvertToPyramids(source, levels)
    _, LT = ConvertToPyramids(target, levels)
    GM, _ = ConvertToPyramids(mask, levels)

    # Use the the Gaussian pyramid to combine the source and target Laplacian pyramids as follows:
    # LC(l) = gpMsk(l) × LS(l) + (1 − gpMsk(l)) × LT (l)
    blended_pyramids = []
    for l in range(len(LS)):
        lcl = GM[levels - l] * LS[l] + (1 - GM[levels - l]) * LT[l]
        blended_pyramids.append(lcl)

    # Collapse the blended pyramid LC to reconsruct the final blended image
    blended = blended_pyramids[0]
    for i in range(1, levels + 1):
        blended = cv2.pyrUp(blended)
        blended = cv2.resize(blended, (blended_pyramids[i].shape[1], blended_pyramids[i].shape[0]))
        blended = blended + blended_pyramids[i]

    #Clamp values to avoid issues of values needing to be 0...1
    return np.clip(blended, 0, 1)

if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # main area to specify files and display blended image

    index = 1

    # Read data and clean mask
    source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

    # Cleaning up the mask
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0

    ### The main part of the code ###

    # Implement the PyramidBlend function (Task 2)
    pyramidOutput = PyramidBlend(source, mask, target)

    # Writing the result

    plt.imsave("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)
