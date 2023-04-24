# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import skimage

# Function to retrieve r, g, b planes from Prokudin-Gorskii glass plate images
def read_strip(path):
    image = plt.imread(path) # read the input image
    info = np.iinfo(image.dtype) # get information about the image type (min max values)
    image = image.astype(np.float64) / info.max # normalize the image into range 0 and 1

    height = int(image.shape[0] / 3)

    # For images with different bit depth
    scalingFactor = 255 if (np.max(image) <= 255) else 65535
    
    # Separating the glass image into R, G, and B channels
    b = image[: height, :]
    g = image[height: 2 * height, :]
    r = image[2 * height: 3 * height, :]
    return r, g, b


# circshift implementation similar to matlab
def circ_shift(channel, shift):
    shifted = np.roll(channel, shift[0], axis = 0)
    shifted = np.roll(shifted, shift[1], axis = 1)
    return shifted


# The main part of the code. Implement the FindShift function
def find_shift(im1, im2):
    '''
    Finds the best shift for an image using exhaustive greedy search

    :param im1: image to shift
    :param im2: image of reference
    :return: the best shift given the parameters
    '''
    best_score = 999999999
    best_shift = [0,0]

    for tx in range(-20,21):         #from -20 to 20
        for ty in range(-20, 21):

            #Shifting with (tx, ty)
            shift = circ_shift(im1, [tx,ty])

            #Checking
            score = np.sum((im2[20:-20, 20:-20] - shift[20:-20, 20:-20])**2)
            if score < best_score:
                best_score = score
                best_shift = [tx, ty]

    return best_shift


def find_pyramid_helper(im1, im2, sr, offset, current_scale):
    best_score = np.inf
    best_shift = [0,0]

    for tx in range(current_scale[0] - sr, current_scale[0] + sr + 1):
        for ty in range(current_scale[1] - sr, current_scale[1] + sr + 1):

            #Shifting with (tx, ty)
            shiftedimg = circ_shift(im1, [tx, ty])

            #Checking
            score = np.sum((shiftedimg[offset:-offset, offset:-offset] - im2[offset:-offset, offset:-offset]) ** 2)

            if score < best_score:
                best_score = score
                best_shift = [tx, ty]

    return best_shift


def find_pyramid(im1, im2):
    current_shift = [0,0]
    pyramid_levels = 4
    offset = 20
    sr = 20

    for i in range(pyramid_levels, 0, -1): #decrementing (aka starting on the smallest image)
        scale = 2 ** (i - 1)

        #Resizing and shifting from current scale
        resized_im1 = skimage.transform.resize(im1, (im1.shape[0] // scale, im1.shape[1] // scale))
        resized_im2 = skimage.transform.resize(im2, (im2.shape[0] // scale, im2.shape[1] // scale))

        resized_im1 = circ_shift(resized_im1, current_shift)
        resized_im2 = circ_shift(resized_im2, current_shift)

        #Finding the best shift given the new image resolutions
        shift = find_pyramid_helper(resized_im1, resized_im2, sr, offset, current_shift)

        #Scaling the shift and metadata
        current_shift = np.array(shift) * 2
        offset *= 2
        sr //= 2

    return shift

if __name__ == '__main__':
    # Setting the input output file path
    imageDir = '../Images/'
    imageName = 'harvesters.tif'
    outDir = '../Results/'

    # Get r, g, b channels from image strip
    r, g, b = read_strip(imageDir + imageName)

    # Calculate shift
    rShift = find_pyramid(r, b)
    gShift = find_pyramid(g, b)
    print(rShift, gShift)

    '''
        JPG
        Catherdal: [12, 3] [5, 2]
        Monastery: [3, 2] [-3, 2]
        Navtivity: [7, 1] [3, 1]
        Settlers: [14, -1] [7, 0]
        
        TIF
        Emir: [105, 40] [49, 24]
        Harvesters: [123, 14] [60, 16]
        Icon: [88, 23] [41, 17]
        Lady: [110, 12] [56, 8]
        Self Portrait: [175, 36] [79, 29]
        Three Generations: [110, 10] [53, 13]
        Train: [86, 30] [42, 3]
        Turkmen: [113, 26] [56, 19]
        Village: [137, 21] [65, 12]
        
    '''

    # Shifting the images using the obtained shift values
    finalB = b
    finalG = circ_shift(g, gShift)
    finalR = circ_shift(r, rShift)

    # Putting together the aligned channels to form the color image
    finalImage = np.stack((finalR, finalG, finalB), axis = 2)

    # Writing the image to the Results folder
    plt.imsave(outDir + imageName[:-4] + '.jpg', finalImage)
