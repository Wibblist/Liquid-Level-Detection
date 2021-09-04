import numpy as np
from math import isclose
import cv2 as cv


def padding(image, kernel):
    k_row, k_col = kernel.shape
    (op_row, op_col) = image.shape

    p_height = int((k_row - 1) / 2)  # padding=(k-1)/2
    p_width = int((k_col - 1) / 2)
    padded_image = np.zeros((op_row + (2 * p_height), op_col + (2 * p_width)))
    padded_image[
        p_height : padded_image.shape[0] - p_height,
        p_width : padded_image.shape[1] - p_width,
    ] = image

    return padded_image.astype(np.uint8)


# calculates normal distribution
def norm(x, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power(x / sigma, 2) / 2)


# create a gaussian kernel
def gauss_kernel(size, sigma=1):  # size of kernel and variance

    # returns an array of (size) number of evenly
    # spaced whole numbers between + and - size/2
    k_1D = np.linspace(-(size // 2), size // 2, size)

    # returns density with mean=0 and variance=sigma
    for i in range(size):
        k_1D[i] = norm(k_1D[i], sigma)

    k_2D = np.outer(k_1D.T, k_1D.T)  # make a 2D with the outer product of the 1D kernel
    k_2D *= 1.0 / k_2D.max()  # normalize and make center value of kernel always 1

    return k_2D  # output 2D kernel


# after image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def convolution(image, kernel):

    (op_row, op_col,) = image.shape
    # setting dimensions of output image the same as the input
    k_row, k_col = kernel.shape

    output = np.zeros(image.shape)  # intializing output as an array of zeros

    padded_image = padding(image, kernel)

    for row in range(op_row):
        for col in range(op_col):
            # apply convolution
            output[row, col] = np.sum(
                kernel * padded_image[row : row + k_row, col : col + k_col]
            )
            # smoothening effect
            output[row, col] /= kernel.shape[0] * kernel.shape[1]
            # current pixel/total kernel pixels

    return output.astype(np.uint8)  # return colvolved image


# apply gaussian blur by creating a gaussian kernel and applying convolution
def gauss_blur(image, kernel_size):
    kernel = gauss_kernel(
        kernel_size, sigma=np.sqrt(kernel_size)
    )  # set standard deviation as square root of kernel size
    return convolution(image, kernel)


# manual threshold
# (T, bottle_threshold) = cv.threshold(bottle_gray, 60, 255, cv.THRESH_BINARY_INV)


def ib_threshold(image, thresh):
    (op_row, op_col) = image.shape
    output = np.zeros(image.shape)

    for row in range(op_row):
        for col in range(op_col):
            # if the current pixel has a higher
            # intensity than the threshold,
            # set to black
            if image[row, col] < thresh:
                output[row, col] = 255
            # else set to white

    return output.astype(np.uint8)


def morph(image, kernel, key=False):

    if key == False:  # erosion
        a = 255
        b = 0
    elif key == True:  # dilation
        a = 0
        b = 255

    row, col = image.shape
    kr, kc = kernel.shape
    output = np.zeros(image.shape)

    for r in range(row):
        for c in range(col):
            k = image[r : r + kr, c : c + kc]
            check = np.all(k == a)

            if check:
                output[r, c] = a
            else:
                output[r, c] = b

    return output.astype(np.uint8)


def find_edge(image, image2, points):

    pnts = []
    row, col = image.shape

    for r in range(row):
        start_found = False
        for c in range(col):
            if (
                (image[r, c] == 255)
                and (image[r - 1, c - 1] == 0)
                and (image[r + 1, c - 1] == 0)
                and (image[r + 1, c] == 0)
            ):
                x = r
                y = c
                start_found = True
                break
        if start_found is True:
            break

    pnts.append((x, y))

    def cw(face):
        if face == "r":
            return "d"
        elif face == "l":
            return "u"
        elif face == "u":
            return "r"
        elif face == "d":
            return "l"

    def ccw(face):
        if face == "r":
            return "u"
        elif face == "l":
            return "d"
        elif face == "u":
            return "l"
        elif face == "d":
            return "r"

    face = "r"
    done = False
    count = 0

    while done == False:
        if face == "r":
            p1 = (x - 1, y + 1)
            p2 = (x, y + 1)
            p3 = (x + 1, y + 1)
            p4 = (x - 1, y)

        elif face == "l":
            p1 = (x + 1, y - 1)
            p2 = (x, y - 1)
            p3 = (x - 1, y - 1)
            p4 = (x + 1, y)

        elif face == "u":
            p1 = (x - 1, y - 1)
            p2 = (x - 1, y)
            p3 = (x - 1, y + 1)
            p4 = (x, y - 1)

        elif face == "d":
            p1 = (x + 1, y + 1)
            p2 = (x + 1, y)
            p3 = (x + 1, y - 1)
            p4 = (x, y + 1)

        if image[p3[0], p3[1]] == 255:
            pnts.append(p2)
            pnts.append(p3)
            x, y = p3
            face = cw(face)
            count = 0
        elif image[p2[0], p2[1]] == 255:
            pnts.append(p2)
            x, y = p2
            count = 0
        elif image[p1[0], p1[1]] == 255:
            pnts.append(p4)
            pnts.append(p1)
            x, y = p1
            count = 0
        else:
            face = ccw(face)
            count += 1

        if ((x, y) == pnts[0]) or count == 3:
            done = True

    pnts.pop()

    pnts.sort()

    for p in pnts:
        image2[p[0], p[1]] = 255

    xl = []  # empty list of rows
    for i in pnts:  # for each edge coord
        if i[0] not in xl:  # add every unique row indexes to xl
            xl.append(i[0])

    # so xl is now a list of every row in the array that has points

    for i in xl:  # for each row that has points
        yl = []  # initialize a list to contain column indexes
        for e in pnts:  # for each edge coord
            if e[0] == i:  # if the point is on the current row
                yl.append(e[1])  # add the column index to yl

        image[i, yl[0] : yl[-1] + 1] = 0
        # turn every pixel in the image from the first point
        # on this row to the last point on the row to black

    points.append(pnts)

    return (image, image2)


def find_edges(image):
    p_k = np.zeros((5, 5))
    image = padding(image, p_k)

    done = False
    cnt = 0
    image2 = np.zeros(image.shape)
    points = []

    while done == False:
        image, image2 = find_edge(image, image2, points)
        cnt += 1
        if cnt > 4:
            break

        if np.all(image == 0):
            done = True

    return image2, points


def draw(image, points):
    def slen(e):
        return len(e)

    points.sort(key=slen)

    liquid = points[-1]
    xp = []
    yp = []
    for i in liquid:
        xp.append(i[0])
        yp.append(i[1])

    cap = points[-2]
    xc = []
    yc = []
    for i in cap:
        xc.append(i[0])
        yc.append(i[1])

    p_top = min(xc)  # top point cap
    p_topL = min(xp)  # top point liquid
    p_bot = max(xp)  # bottom point liquid
    p_left = min(yp)  # left most point liquid
    p_right = max(yp)  # right most point liquid

    w = p_right - p_left
    h = p_bot - p_topL

    red = [255, 0, 0]
    green = [0, 160, 0]

    font = cv.FONT_HERSHEY_SIMPLEX
    text = "Amount Remaining: " + str(round(100 * ((h / w) / 2.75), 1)) + "%"

    pos = (int((image.shape[1] / 2) - 310), 100)

    if ((h / w) / 2.75) > 0.79999:

        if isclose(h / w, 2.75, abs_tol=0.01) or ((h / w) / 2.75) > 1:
            image = cv.putText(
                image, "Amount Remaining: 100%", pos, font, 1.5, green, 2
            )
        else:
            image = cv.putText(image, text, pos, font, 1.5, green, 2)

        a = p_top
        while a < p_bot:
            image[a, p_left] = green
            image[a, p_left - 1] = green
            image[a, p_left + 1] = green
            image[a, p_left - 2] = green
            image[a, p_left + 2] = green
            a += 1

        a = p_left
        while a < p_right:
            image[p_top, a] = green
            image[p_top - 1, a] = green
            image[p_top + 1, a] = green
            image[p_top - 2, a] = green
            image[p_top + 2, a] = green
            a += 1

        a = p_top
        while a < p_bot:
            image[a, p_right] = green
            image[a, p_right - 1] = green
            image[a, p_right + 1] = green
            image[a, p_right - 2] = green
            image[a, p_right + 2] = green
            a += 1

        a = p_left
        while a < p_right:
            image[p_bot, a] = green
            image[p_bot - 1, a] = green
            image[p_bot + 1, a] = green
            image[p_bot - 2, a] = green
            image[p_bot + 2, a] = green
            a += 1

    else:
        image = cv.putText(image, text, pos, font, 1.5, red, 2)

    return image
