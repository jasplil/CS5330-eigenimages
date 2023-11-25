# Bruce Maxwell
# Modified by: Yixiang Xie
# CS 5330 F23
# Tool for building a set of eigenimages from a collection of images
#
#
# 1: run the code and make sure it works, you may need to install openCV with pip
# 2: pick a set of images of your own that are coherent in some way
# 3: crop the images according to some standard of your own
# 4: put the images into a new directory
# 5: run the program with the new directory as the argument
# 6: modify the code so (after the existing code) it reads a new image (not in the directory) and does the following
#    A) display the original image
#    B) subtract the mean image from the original image to get the differential image
#    C) display the differential image
#    D) project the differential image into eigenspace by taking the dot product with the first N eigenvectors
#    E) print the projected coefficients
#    F) re-project from the coefficients back to a differential image
#    G) display the re-projected differential image
#    H) add the mean image to the re-projected differential image to get the re-projected original
#    I) display the re-projected original image
#
# Try step six with an image that is similar to the ones you used to create the space
# Try step six with an image that is not similar to the ones you used to create the space

import sys
import os
import numpy as np
import cv2


# make the image into a single channel, and then resize and reshape it
def imagePreprocess(image, rows, cols):
    # make the image a single channel
    image = image[:, :, 1]

    # resize the image to the given size
    image = cv2.resize(image, (cols, rows), interpolation = cv2.INTER_AREA)

    # reshape the image to a 1-D single vector
    image = np.reshape(image, image.shape[0] * image.shape[1])
    image = image.astype("float32")

    return image


# reshapes and normalizes a 1-D column image,
# works for eigenvectors, differential images, and original images
def viewColumnImage(column, rows, cols, name):
    # reshape the image to a 2-D array
    image = np.reshape(column, (rows, cols))

    # normalize the image to 0-255
    minVal = np.min(image)
    maxVal = np.max(image)
    image = 255 * ((image - minVal) / (maxVal - minVal))
    view = image.astype("uint8")

    cv2.imshow(name, view)
    cv2.imwrite(name + ".jpg", view)
    

def main(argv):
    # check for a directory path
    if len(argv) < 4:
        print("usage: python %s <directory path> <similar test image> <not similar test image>" % (argv[0]))
        return

    # grab the directory path and test image paths
    imageDir = argv[1]
    similarImagePath = argv[2]
    notSimilarImagePath = argv[3]

    # open the directory and get a file listing
    filelist = os.listdir(imageDir)

    print("Processing image dir")

    # the resize image size
    rows = 0
    cols = 0

    firstImage = True
    for filename in filelist:
        print("Processing file %s" % (filename))

        # skip non image files
        suffix = filename.split(".")[-1]
        if not ('pgm' in suffix or 'tif' in suffix or 'jpg' in suffix or 'png' in suffix):
            continue

        # read the image
        src = cv2.imread(imageDir + "/" + filename)

        # resize the long side of the image to 160
        if firstImage:
            resizeFactor =  160 / max(src.shape[0], src.shape[1])
            rows = int(src.shape[0] * resizeFactor)
            cols = int(src.shape[1] * resizeFactor)

        image = imagePreprocess(src, rows, cols)

        # could probably normalize the image to sum to one
        # image /=  np.sum(image)

        if firstImage:
            aMtx = image
            firstImage = False
        else:
            aMtx = np.vstack((aMtx, image))

    # all of the images are in aMtx, which has num of images rows and image rows * cols columns
    # compute mean image
    meanVec = np.mean(aMtx, axis = 0)

    # look at the mean vector
    viewColumnImage(meanVec, rows, cols, "Mean image")
    cv2.moveWindow("Mean image", 0, 0)

    # build the differential data matrix and transpose so each image is a column
    dMtx = aMtx - meanVec
    dMtx = dMtx.T

    # compute the singular value decomposition
    # N images
    #
    # U is rows*cols x N, columns of U are the eigenvectors of dMtx
    # V is N x N, rows of V would be the eigenvectors of the rows of dMtx
    # s contains the singular values which are related to the eigenvalues
    U, s, V = np.linalg.svd(dMtx, full_matrices = False)

    # print the top 20 eigenvalues
    eval = s**2 / (dMtx.shape[0] - 1)
    print("Top 20 eigenvalues: ", eval[0:20])

    # show the top 8 eigenvectors
    for i in range(8):
        name = "Eigenvector %d" % (i)
        viewColumnImage(U[:,i], rows, cols, name)
        cv2.moveWindow(name, i * 170, 200)

    similarImage = cv2.imread(similarImagePath)
    similarImage = imagePreprocess(similarImage, rows, cols)
    notSimilarImage = cv2.imread(notSimilarImagePath)
    notSimilarImage = imagePreprocess(notSimilarImage, rows, cols)

    # project the test images onto the first fifteen eigenvectors
    # and then re-project back to the original space
    # and visualize the results
    position = 0
    for index, image in enumerate([similarImage, notSimilarImage]):
        name = "Similar" if index == 0 else "Not Similar"

        # display the original image
        viewColumnImage(image, rows, cols, "Original " + name)
        cv2.moveWindow("Original " + name, position * 170, 400)
        
        # get the differential image
        image = image - meanVec

        # show the differential image
        viewColumnImage(image, rows, cols, "Differential " + name)
        cv2.moveWindow("Differential " + name, (position + 1) * 170, 400)

        # take the dot product of the differential image and the first fifteen eigenvectors
        projection = np.dot(image.T, U[:,0:15])

        # print the coefficients
        toPrint = name + " Coefficients: "
        for j in range(len(projection)):
            toPrint +=  "%7.1f  " % (projection[j])
        print(toPrint)

        # re-project from the fifteen coefficients back to a new image
        recreated = projection[0] * U[:,0]
        for j in range(1,len(projection)):
            recreated +=  projection[j] * U[:,j]  # sum the coefficients times the corresponding eigenvectors

        # show the recreated original image (after adding back the mean image) note less noise
        viewColumnImage(recreated + meanVec, rows, cols, "Recreated Original " + name)
        cv2.moveWindow("Recreated Original " + name, position * 170, 600)

        # show the recreated differential image, note less noise
        viewColumnImage(recreated, rows, cols, "Recreated " + name)
        cv2.moveWindow("Recreated " + name, (position + 1) * 170, 600)

        position +=  2

    cv2.waitKey(0)
    

if __name__ ==  "__main__":
    main(sys.argv)
