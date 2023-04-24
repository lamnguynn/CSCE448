# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import spdiags, csr_matrix, linalg, lil_matrix, diags, block_diag


# Read source, target and mask for a given id
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


# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset
    
    if (xOffset < 0):
        mask    = mask[abs(xOffset):, :]
        source  = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask    = mask[:, abs(yOffset):]
        source  = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask    = mask[:sourceHeight, :]
        source  = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask    = mask[:, :sourceWidth]
        source  = source[:, :sourceWidth]
    
    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal


# Pyramid Blend
def PyramidBlend(source, mask, target):
    
    return source * mask + target * (1 - mask)


# Poisson Blend

def PoissonBlend(source, mask, target, isMix):
    # Get dimensions
    height, width, channel = target.shape
    num_pixels = height * width

    # Compute matrix A and vector b

    # Loop through the mask to find A. We will also precompute its content since traversing it is slow due to being linked list
    A_data, A_row, A_col = [], [], []
    for y in range(height):
        for x in range(width):
            i = y * width + x
            if (mask[y, x] == 0).any():
                A_data.append(1)
                A_row.append(i)
                A_col.append(i)
            else:
                A_data.append(4)
                A_row.append(i)
                A_col.append(i)
                if x > 0:               #A[i, i - 1] = -1
                    A_data.append(-1)
                    A_row.append(i)
                    A_col.append(i - 1)
                if x < width - 1:       #A[i, i + 1] = -1
                    A_data.append(-1)
                    A_row.append(i)
                    A_col.append(i + 1)
                if y > 0:               #A[i, i - width] = -1
                    A_data.append(-1)
                    A_row.append(i)
                    A_col.append(i - width)
                if y < height - 1:      #A[i, i + width] = -1
                    A_data.append(-1)
                    A_row.append(i)
                    A_col.append(i + width)

    # Construct sparse matrix A
    A = csr_matrix((A_data, (A_row, A_col)), shape=(num_pixels, num_pixels))

    # Loop through each channel to find b respective to the channel
    blend = np.zeros(target.shape)
    for c in range(channel):
        b = np.zeros((num_pixels))
        for y in range(height):
            for x in range(width):
                i = y * width + x

                if (mask[y,  x] == 0).any():
                    b[i] = target[y, x, c]
                else:
                    # use derived equation to fill in b. also need to check bounds to avoid errors
                    b[i] = 4 * source[y, x, c]
                    if x > 0:
                        b[i] -= source[y, x - 1, c]
                    if y > 0:
                        b[i] -= source[y - 1, x, c]
                    if x < width - 1:
                        b[i] -= source[y, x + 1, c]
                    if y < height - 1:
                        b[i] -= source[y + 1, x, c]

        # Solve Ax = b
        x = linalg.spsolve(A, b)

        # Reshape x to target size
        blend[:, :, c] = np.reshape(x, target[:, :, c].shape)

    return np.clip(blend, 0, 1)


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'
    
    # False for source gradient, true for mixing gradients
    isMix = False

    # Source offsets in target
    offsets = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88]]

    # main area to specify files and display blended image
    for index in range(len(offsets)):
        # Read data and clean mask
        source, maskOriginal, target = Read(str(index+1).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])

        ### The main part of the code ###
    
        # Implement the PoissonBlend function

        poissonOutput = PoissonBlend(source, mask, target, isMix)

        
        # Writing the result
                
        if not isMix:
            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
        else:
            plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
