import numpy as np
import cv2
import math
from scipy.spatial import distance
import pickle


def detectAsymmetry(img, cnt):
    distance_ratio = 0
    c = cnt

    perimeter = cv2.arcLength(c, True)

    # Obtaining the top, bottem, left and right points of the lesion
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBottom = tuple(c[c[:, :, 1].argmax()][0])

    # Drawing the contour and the top, bottem, left and right points of the lesion
    cv2.drawContours(img, c, -1, (0, 255, 255), 2)
    cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(img, extRight, 8, (0, 255, 0), -1)
    cv2.circle(img, extTop, 8, (255, 0, 0), -1)
    cv2.circle(img, extBottom, 8, (255, 255, 0), -1)

    # Obtaining the distance between given two points in the XY cordinate plane.
    # Proven that it is equal to the pythogorous theorem
    dist1 = math.sqrt(((extTop[0] - extBottom[0]) ** 2) + ((extTop[1] - extBottom[1]) ** 2))
    dist2 = math.sqrt(((extLeft[0] - extRight[0]) ** 2) + ((extLeft[1] - extRight[1]) ** 2))

    distance_diff = abs(dist1 - dist2)
    if (dist1 == 0 and dist2 == 0):
        distance_ratio = 0
    else:
        distance_ratio = (distance_diff / (dist1 + dist2)) * 100

    # Obtaining the center point of the contour
    M = cv2.moments(c)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    contour_center = (center_X, center_Y)

    # Obtaining the distance from the center of the lesion to the top, bottem, left and right points of the lesion
    x = abs(distance.euclidean(extRight, contour_center) - distance.euclidean(extLeft, contour_center))
    y = abs(distance.euclidean(extTop, contour_center) - distance.euclidean(extBottom, contour_center))

    if (distance_ratio > 5 or x > 25 or y > 25):
        if (x <= 10 or y <= 10):
            return (0, round(distance_ratio, 4), perimeter)
        else:
            return (1, round(distance_ratio, 4), perimeter)
    else:
        return (0, round(distance_ratio, 4), perimeter)


def detectBorderIrregularity(cnt, hull):
    comp = 0

    if len(cnt) > 4:
        ellipse = cv2.fitEllipse(cnt)
        ellipse_cnt = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                       (int(ellipse[1][0]), int(ellipse[1][1])), int(ellipse[2]), 0, 360, 1)
        # Check the difference between given two contours to check for the irregularity
        compOri = cv2.matchShapes(cnt, ellipse_cnt, 1, 0.0)
        compNew = cv2.matchShapes(cnt, hull, 1, 0.0)

    if (compNew > 0.3):
        return (1, round(compOri, 4), round(compNew, 4))
    else:
        return (0, round(compOri, 4), round(compNew, 4))


def detectColour(img, cnt, state):
    hh, ww = img.shape[:2]
    count = 0

    if (state == 1):
        c = scale_contour(cnt, 0.85)

        mask3 = np.zeros((hh, ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(img, img, mask=mask3)

    else:
        result = img

    lower_brown1 = np.array([0, 120, 0])
    upper_brown1 = np.array([90, 150, 190])
    lower_brown2 = np.array([135, 120, 0])
    upper_brown2 = np.array([180, 150, 100])

    lower_red = np.array([160, 130, 150])
    upper_red = np.array([210, 255, 255])

    lower_blue = np.array([94, 70, 80])
    upper_blue = np.array([130, 255, 255])

    lower_white = np.array([0, 0, 100])
    upper_white = np.array([225, 90, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])

    allArea = 0
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        allArea += cv2.contourArea(c)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, lower_black, upper_black)
    res = cv2.bitwise_and(result, result, mask=mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0
    for c in contours:
        selectedArea += cv2.contourArea(c)
    percentage = selectedArea / allArea * 100
    if (percentage > 5):
        count += 1
        black = 1
    else:
        black = 0

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(result, result, mask=mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0
    for c in contours:
        selectedArea += cv2.contourArea(c)
    percentage = selectedArea / allArea * 100
    if (percentage > 5):
        count += 1
        white = 1
    else:
        white = 0

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(result, result, mask=mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0
    for c in contours:
        selectedArea += cv2.contourArea(c)
    percentage = selectedArea / allArea * 100
    if (percentage > 5):
        count += 1
        blue = 1
    else:
        blue = 0

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(result, result, mask=mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0
    for c in contours:
        selectedArea += cv2.contourArea(c)
    percentage = selectedArea / allArea * 100
    if (percentage > 5):
        count += 1
        red = 1
    else:
        red = 0

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    mask = mask1 + mask2
    res = cv2.bitwise_and(result, result, mask=mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selectedArea = 0
    for c in contours:
        selectedArea += cv2.contourArea(c)
    percentage = selectedArea / allArea * 100
    if (percentage > 5):
        count += 1
        brown = 1
    else:
        brown = 0

    if (count > 2):
        return (brown, red, blue, white, black, 1, count)
    else:
        return (brown, red, blue, white, black, 0, count)


def detectDiameter(img, cnt):
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)

    diameter = 2 * radius / 100 * 2
    # Obtaining the rectangle that could cover the entire contour
    x, y, w, h = cv2.boundingRect(cnt)

    if (h >= w):
        d = h
    else:
        d = w

    d = d * 25.4 / 1280

    if (d > 6):
        return (1, round(d, 4), w, h)
    else:
        return (0, round(d, 4), w, h)


def detectGlobules(img, cnt):
    params = cv2.SimpleBlobDetector_Params()

    # This parameter is being used to say about the size of the blob
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 500

    # This paramater is being used to say about the circularity of the blob where 1 is a circle
    params.filterByCircularity = True
    params.minCircularity = 0.2

    # This parameter is being used to say how close its shape seems to be like a circle where 1 is a circle
    params.filterByConvexity = True
    params.minConvexity = 0.3

    # This paramater is being used to say how close its shape seems to be like a line
    # where 0 is a line and 1 is a circle. Intermediate values are to show ellipses
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # This paramater is being used to say about the colour that we expect the blob to be where 0 is dark and 255 is light
    params.filterByColor = True
    params.blobColor = 0

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)

    number_of_blobs = len(keypoints)

    if (number_of_blobs > 15):
        return (1)
    else:
        return (0)


def detectBlotches(img, cnt):
    # Obtaining the image and applying GaussianBlur to smooth the contour edges
    img = cv2.GaussianBlur(img, (5, 5), 0)
    src3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernal = np.ones((7, 7), np.uint8)
    # Filling the points inside the foreground objects
    src2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal)
    # src1 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)

    src1 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    # The thresholding values are being set to detect the lesion clearly
    # If the pixel value is lesser than the threhold the the pixel will be set to 0. Can only work grayscale images
    # Creating a mask to crop the lesion in the image
    # Obtaing the lesion from the image using created mask
    ret, thresh2 = cv2.threshold(src1, 65, 255, cv2.THRESH_BINARY_INV)
    mask3 = cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB)
    im_thresh_color = cv2.bitwise_and(src3, mask3)

    # The thresholding values are being set to detect the lesion clearly
    # If the pixel value is lesser than the threhold the the pixel will be set to 0. Can only work grayscale images
    # Creating a mask to crop the lesion in the image
    # Obtaing the lesion from the image using created mask
    retOther, threshOther = cv2.threshold(src1, 125, 255, cv2.THRESH_BINARY_INV)
    maskOther = cv2.cvtColor(threshOther, cv2.COLOR_BGR2RGB)
    im_thresh_color_other = cv2.bitwise_and(src3, maskOther)

    # Obtaining the larger area from the contour
    gray = cv2.cvtColor(im_thresh_color_other, cv2.COLOR_BGR2GRAY)
    # Get the external contours not the entire hierarchy of contours (unlike TREE)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    allArea = 0
    if (len(contours) != 0):
        c = max(contours, key=cv2.contourArea)
        c = scale_contour(c, 0.8)
        allArea = cv2.contourArea(c)

    # Obtaining the larger area from the contour
    gray = cv2.cvtColor(im_thresh_color, cv2.COLOR_BGR2GRAY)
    # Get the external contours not the entire hierarchy of contours (unlike TREE)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blotchArea = 0
    if (len(contours) != 0):
        c = max(contours, key=cv2.contourArea)
        blotchArea = cv2.contourArea(c)
    percentage = 0

    if (allArea != 0):
        percentage = blotchArea / allArea * 100

    if (percentage > 30 and percentage < 90):
        return (1)
    else:
        return (0)


def detectMilkyRedAreas(img, cnt):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([170, 140, 140])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    output_img = img.copy()
    output_img[np.where(mask1 == 0)] = 0

    # Check whether the red area is very small in the code

    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask1 == 0)] = 0

    gray = cv2.cvtColor(output_hsv, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    output_red = img.copy()
    result = cv2.bitwise_and(output_red, output_red, mask=thresh);

    allArea = np.sum(output_img != 0)
    selectedArea = np.sum(result != 0)

    if (allArea == 0):
        return (0)
    else:
        percentage = selectedArea / allArea * 100
        if (percentage > 15):
            return (1)
        else:
            return (0)


def detectRosettes(img, cnt, state):
    hh, ww = img.shape[:2]
    if (state == 1):
        c = scale_contour(cnt, 0.7)

        mask3 = np.zeros((hh, ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(img, img, mask=mask3)

    else:
        result = img

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.equalizeHist(gray)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 100

    params.filterByCircularity = True
    params.minCircularity = 0.6

    params.filterByConvexity = True
    params.minConvexity = 0.8

    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(result)

    if (len(keypoints) > 4):
        return (1)
    else:
        return (0)


def detectRegressionStructure(img, cnt, state):
    hh, ww = img.shape[:2]
    if (state == 1):
        c = scale_contour(cnt, 0.6)

        mask3 = np.zeros((hh, ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(img, img, mask=mask3)

    else:
        result = img

    lower_white = np.array([0, 0, 145])
    upper_white = np.array([175, 123, 255])

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(result, result, mask=mask2)

    allArea = np.sum(result != 0)
    selectedArea = np.sum(res != 0)

    if (allArea == 0):
        return (0)
    else:
        percentage = selectedArea / allArea * 100
        if (percentage > 60):
            return (1)
        else:
            return (0)


def detectBlueWhiteVeil(img, cnt, state):
    hh, ww = img.shape[:2]
    if (state == 1):
        c = scale_contour(cnt, 0.85)

        mask3 = np.zeros((hh, ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(img, img, mask=mask3)

    else:
        result = img

    # Defining the colour range that we are looking for. in the lesion
    lower_bluegray = np.array([94, 80, 2])
    upper_bluegray = np.array([126, 255, 255])

    # Converting the image to the HSV format
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    # Creating a mask from the area that the given colour appears in the lesion
    mask2 = cv2.inRange(hsv, lower_bluegray, upper_bluegray)
    # Cropping the image using the mask
    res = cv2.bitwise_and(result, result, mask=mask2)

    # Get the area of the contour and the area taken by the considered colour
    allArea = np.sum(result != 0)
    selectedArea = np.sum(res != 0)

    if (allArea == 0):
        return (0)
    else:
        percentage = selectedArea / allArea * 100
        if (percentage > 15):
            return (1)
        else:
            return (0)


def detectAtypicalNetwork(img, cnt, state):
    hh, ww = img.shape[:2]
    if (state == 1):
        c = scale_contour(cnt, 0.85)

        # Creating a mask to crop the lesion in the image
        mask3 = np.zeros((hh, ww), dtype=np.uint8)
        mask3 = cv2.drawContours(mask3, [c], 0, (255, 255, 255), cv2.FILLED)
        # Obtaing the lesion from the image using created mask
        result = cv2.bitwise_and(img, img, mask=mask3)

    else:
        result = img

    # Obtaining the area of the lesion
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    allArea = cv2.contourArea(c)

    # Obtaining the area where the atypical network is covering
    grayScale = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # contours,hierarchy = cv2.findContours(blackhat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Detect the blobs in the based on the given criteria
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 50

    params.filterByCircularity = True
    params.minCircularity = 0.5

    params.filterByConvexity = True
    params.minConvexity = 0.01

    params.filterByInertia = False
    params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh2)

    # Obtaining the pixel areas of black and white
    number_of_white_pix = np.sum(thresh2 == 255)
    percentage = number_of_white_pix / allArea * 100

    if (len(keypoints) > 15 and percentage > 30):
        return (1)
    else:
        return (0)


def detectStreaks(img, cnt):
    hh, ww = img.shape[:2]
    c1 = scale_contour(cnt, 1.1)
    c = scale_contour(cnt, 0.75)

    maskSmall = np.zeros((hh, ww), dtype=np.uint8)
    maskSmall = cv2.drawContours(maskSmall, [c], 0, (255, 255, 255), cv2.FILLED)

    maskLarge = np.zeros((hh, ww), dtype=np.uint8)
    maskLarge = cv2.drawContours(maskLarge, [c1], 0, (255, 255, 255), cv2.FILLED)

    result2 = cv2.bitwise_and(img, img, mask=maskLarge - maskSmall);

    hsv = cv2.cvtColor(result2, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    masknew = cv2.inRange(hsv, lower_black, upper_black)
    res = cv2.bitwise_and(result2, result2, mask=masknew)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    allArea = 0

    for c in contours:
        allArea += cv2.contourArea(c)
        cv2.drawContours(result2, c, -1, (0, 0, 255), 2)

    if (allArea > 0):
        grayScale = cv2.cvtColor(result2, cv2.COLOR_RGB2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

        ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        invert = cv2.bitwise_not(cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB))

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 200

        params.filterByCircularity = True
        params.minCircularity = 0.01

        params.filterByConvexity = True
        params.minConvexity = 0.4

        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(invert)

        number_of_black_pix = np.sum(invert == 0)

        keys = 0
        for k in keypoints:
            keys += (k.size * k.size * 3.14 * 0.25)
        non_black = np.sum(result2 != 0)

        key_percentage = keys / number_of_black_pix * 100
        contour_percentage = allArea / non_black * 100
        total = key_percentage + contour_percentage

        if (total > 25 or (contour_percentage > 0.4 and key_percentage > 6 and total > 12) or (
                total > 12 and len(keypoints) > 10)):
            return (1)
        else:
            return (0)
    else:
        return (0)


def removeHair(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Changing the shape and the size of the kernal
    # Kernal shows how any pixel in an image combines with different amounts of neigbouring pixels
    kernel = cv2.getStructuringElement(1, (5, 5))

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    ret, thresh2 = cv2.threshold(blackhat, 100, 255, cv2.THRESH_BINARY)

    # Filling the gaps that were taken by the pixels that showed hair by mixing up with the nearby pixel colours
    dst = cv2.inpaint(img, thresh2, 15, cv2.INPAINT_TELEA)

    return dst


def removeLens(img):
    copy = img

    kernel = np.ones((9, 9), np.uint8)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = 255 - mask
    img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, channels = img.shape

    border_threshold_R = 4
    border_threshold_G = 4
    border_threshold_B = 4

    borderup = 0
    borderdown = 0
    borderleft = 0
    borderright = 0

    upone = 0
    downone = 0
    leftone = 0
    rightone = 0

    upthree = 0
    downthree = 0
    leftthree = 0
    rightthree = 0

    # Checking whether the top border exists in the middle of the image
    for i in range(int(height / 2)):
        mid_pixel_top_half = img[i][int(width / 2)]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            borderup += 1
        else:
            break

    # Checking whether the bottom border exists in the middle of the image
    for i in range(height - 1, int(height / 2) - 1, -1):
        mid_pixel_bottom_half = img[i][int(width / 2)]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            borderdown += 1
        else:
            break

    # Checking whether the left border exists in the middle of the image
    for i in range(int(width / 2)):
        mid_pixel_top_half = img[int(height / 2)][i]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            borderleft += 1
        else:
            break

    # Checking whether the right border exists in the middle of the image
    for i in range(width - 1, int(width / 2) - 1, -1):
        mid_pixel_bottom_half = img[int(height / 2)][i]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            borderright += 1
        else:
            break

        #############################

    # Checking whether the top border exists in the first quater of the image
    for i in range(int(height / 2)):
        mid_pixel_top_half = img[i][int(width / 4)]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            upone += 1
        else:
            break

    # Checking whether the bottom border exists in the first quater of the image
    for i in range(height - 1, int(height / 2) - 1, -1):
        mid_pixel_bottom_half = img[i][int(width / 4)]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            downone += 1
        else:
            break

    # Checking whether the left border exists in the first quater of the image
    for i in range(int(width / 2)):
        mid_pixel_top_half = img[int(height / 4)][i]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            leftone += 1
        else:
            break

    # Checking whether the right border exists in the first quater of the image
    for i in range(width - 1, int(width / 2) - 1, -1):
        mid_pixel_bottom_half = img[int(height / 4)][i]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            rightone += 1
        else:
            break

        #############################

    # Checking whether the top border exists in the last quater of the image
    for i in range(int(height / 2)):
        mid_pixel_top_half = img[i][int(3 * width / 4) - 1]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            upthree += 1
        else:
            break

    # Checking whether the bottom border exists in the last quater of the image
    for i in range(height - 1, int(height / 2) - 1, -1):
        mid_pixel_bottom_half = img[i][int(3 * width / 4) - 1]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            downthree += 1
        else:
            break

    # Checking whether the left border exists in the last quater of the image
    for i in range(int(width / 2)):
        mid_pixel_top_half = img[int(3 * height / 4) - 1][i]
        R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            leftthree += 1
        else:
            break

    # Checking whether the right border exists in the last quater of the image
    for i in range(width - 1, int(width / 2) - 1, -1):
        mid_pixel_bottom_half = img[int(3 * height / 4) - 1][i]
        R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
        if (R < border_threshold_R) and (G < border_threshold_G) and (B < border_threshold_B):
            rightthree += 1
        else:
            break

    count = 0
    finalup = 0
    finaldown = 0
    finalleft = 0
    finalright = 0

    if (upone > 0 and upthree > 0 and borderup > 0):
        count += 1
        listVal = [borderup, upone, upthree]
        finalup = max(listVal)

    if (downone > 0 and downthree > 0 and borderdown > 0):
        count += 1
        listVal = [borderdown, downone, downthree]
        finaldown = max(listVal)

    if (rightone > 0 and rightthree > 0 and borderright > 0):
        count += 1
        listVal = [borderright, rightone, rightthree]
        finalright = max(listVal)

    if (leftone > 0 and leftthree > 0 and borderleft > 0):
        count += 1
        listVal = [borderleft, leftone, leftthree]
        finalleft = max(listVal)

    if (count > 0):
        subimage = copy[finalup + 10: height - finaldown - 10, finalleft + 10: width - finalright - 10]
        return subimage
    else:
        return copy


def removeInkPatches(img):
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lowerValues1 = np.array([110, 50, 70])
    lowerValues2 = np.array([110, 100, 120])
    upperValues = np.array([138, 255, 255])

    bluepenMask1 = cv2.inRange(hsvImage, lowerValues1, upperValues)
    bluepenMask2 = cv2.inRange(hsvImage, lowerValues2, upperValues)

    non_black1 = np.sum(bluepenMask1 != 0)
    non_black2 = np.sum(bluepenMask2 != 0)
    percentage = 0
    if (non_black1 != 0):
        percentage = non_black2 / non_black1 * 100
    flags = cv2.INPAINT_TELEA
    if (percentage > 50):
        output = cv2.inpaint(img, bluepenMask1 + bluepenMask2, 100, flags=flags)
    else:
        output = img
    return output


def getContour(img):
    hh, ww = img.shape[:2]
    border = False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The below threshold values works accurately for most of the images. But not for all.
    # Therefore initially they will be checked in order to minimize the excution time and
    # the computational power required to obtain the proper contour that would best fit the lesion in the image

    blur = cv2.GaussianBlur(gray, (17, 17), 32)
    ret, threshBlurred = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contoursBlurred, hierarchy = cv2.findContours(threshBlurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ret, threshNormal = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contoursNormal, hierarchy = cv2.findContours(threshNormal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting the contours based on their sizes in the ascending order
    cntsSorted = sorted(contoursBlurred, key=lambda x: cv2.contourArea(x))
    i = len(cntsSorted) - 1
    if (len(cntsSorted) > 0):
        status = True
        valid = True
        while status:
            if (i < 0):
                status = False
                valid = False
                break
            ca = cntsSorted[i]
            count = 0
            # Filters the contours that are touching the border of the image
            for a in range(len(ca)):
                if ((ww - 5) < ca[a][0][0]):
                    count += 1
            for a in range(len(ca)):
                if ((hh - 5) < ca[a][0][1]):
                    count += 1
            if (0 in ca) or (count != 0):
                i -= 1
            else:
                status = False

        if (valid):
            maskBlurred = np.zeros((hh, ww), dtype=np.uint8)
            maskBlurred = cv2.drawContours(maskBlurred, [ca], 0, (255, 255, 255), cv2.FILLED)
            blurredArea = np.sum(maskBlurred != 0)
            blurredCon = ca
        else:
            maskBlurred = np.zeros((hh, ww), dtype=np.uint8)
            blurredArea = 0
            blurredCon = cntsSorted[len(cntsSorted) - 1]

        # Obtains the largest contour detected even though it may touch the border of the image

        maskBlurredLarge = np.zeros((hh, ww), dtype=np.uint8)
        maskBlurredLarge = cv2.drawContours(maskBlurredLarge, [cntsSorted[len(cntsSorted) - 1]], 0, (255, 255, 255),
                                            cv2.FILLED)
        blurredLargeArea = np.sum(maskBlurredLarge != 0)
        blurredLargeCon = cntsSorted[len(cntsSorted) - 1]
        percentage = blurredLargeArea / (hh * ww) * 100
        x, y, w, h = cv2.boundingRect(cntsSorted[len(cntsSorted) - 1])
        coord = []
        coord.append((0, 0))
        coord.append((0, hh))
        coord.append((ww, 0))
        coord.append((ww, hh))
        countCor = 0
        for cor in coord:
            if (cor == (x, y) or cor == (x + w, y) or cor == (x, y + h) or cor == (x + w, y + h)):
                countCor += 1

        if (percentage > 25 and countCor == 0) or (percentage > 60):
            selected = maskBlurredLarge
            selectedArea = blurredLargeArea
            selectedCon = blurredLargeCon
            border = True
        else:
            selected = maskBlurred
            selectedArea = blurredArea
            selectedCon = blurredCon

    else:
        selected = np.zeros((hh, ww), dtype=np.uint8)
        selectedArea = 0

    cntsSorted = sorted(contoursNormal, key=lambda x: cv2.contourArea(x))
    i = len(cntsSorted) - 1
    if (len(cntsSorted) > 0):
        status = True
        valid = True
        while status:
            if (i < 0):
                status = False
                valid = False
                break
            ca = cntsSorted[i]
            count = 0
            # Filters the contours that are touching the border of the image
            for a in range(len(ca)):
                if ((ww - 5) < ca[a][0][0]):
                    count += 1
            for a in range(len(ca)):
                if ((hh - 5) < ca[a][0][1]):
                    count += 1
            if (0 in ca) or (count != 0):
                i -= 1
            else:
                status = False
        if (valid):
            maskNormal = np.zeros((hh, ww), dtype=np.uint8)
            maskNormal = cv2.drawContours(maskNormal, [ca], 0, (255, 255, 255), cv2.FILLED)
            normalArea = np.sum(maskNormal != 0)
            normalCon = ca
        else:
            normalArea = 0
            normalCon = cntsSorted[len(cntsSorted) - 1]
    else:
        normalArea = 0

    if ((selectedArea < normalArea) or (border and (selectedArea / 2) < normalArea)):
        selected = maskNormal
        selectedCon = normalCon
        selectedArea = normalArea

    src = img
    src = cv2.GaussianBlur(src, (5, 5), 0)
    src3 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    kernal = np.ones((7, 7), np.uint8)
    src2 = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernal)
    src1 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
    retOther, threshOther = cv2.threshold(src1, 125, 255, cv2.THRESH_BINARY_INV)
    maskOther = cv2.cvtColor(threshOther, cv2.COLOR_BGR2RGB)
    im_thresh_color_other = cv2.bitwise_and(src3, maskOther)
    gray = cv2.cvtColor(im_thresh_color_other, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    i = len(cntsSorted) - 1
    if (len(cntsSorted) > 0):
        status = True
        valid = True
        while status:
            if (i < 0):
                status = False
                valid = False
                break
            ca = cntsSorted[i]
            count = 0
            # Filters the contours that are touching the border of the image
            for a in range(len(ca)):
                if ((ww - 5) < ca[a][0][0]):
                    count += 1
            for a in range(len(ca)):
                if ((hh - 5) < ca[a][0][1]):
                    count += 1
            if (0 in ca) or (count != 0):
                i -= 1
            else:
                status = False

    else:
        blotchArea = 0

    maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
    maskBlotch = cv2.drawContours(maskBlotch, [ca], 0, (255, 255, 255), cv2.FILLED)
    blotchArea = np.sum(maskBlotch != 0)
    blotchCon = ca

    if ((selectedArea < blotchArea) or (border and (selectedArea / 2) < blotchArea)):
        selected = maskBlotch
        selectedCon = blotchCon
        selectedArea = blotchArea

    i = img
    result = cv2.bitwise_and(i, i, mask=selected)
    percentage = selectedArea / (hh * ww) * 100

    maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
    hull = cv2.convexHull(selectedCon)
    maskBlotch = cv2.drawContours(maskBlotch, [hull], 0, (255, 255, 255), cv2.FILLED)
    hullResult = cv2.bitwise_and(i, i, mask=maskBlotch)

    # Obtaining the center of the image
    image_center = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape) / 2
    value = image_center[1]
    image_center[1] = image_center[0]
    image_center[0] = value
    image_center = tuple(image_center.astype('int32'))
    # Obtaining the center of the contour
    M = cv2.moments(selectedCon)
    center_X = int(M["m10"] / M["m00"])
    center_Y = int(M["m01"] / M["m00"])
    contour_center = (center_X, center_Y)
    # Below calculated value is the distance between the center of the contour and the center of the image
    distance_to_center = (distance.euclidean(image_center, contour_center))

    # This is used to check whether the contour touches the corners of the image
    corners = 0
    for a in range(len(selectedCon)):
        if (0 == selectedCon[a][0][0] and 0 == selectedCon[a][0][1]):
            corners += 1
        if (0 == selectedCon[a][0][0] and (hh - 1) == selectedCon[a][0][1]):
            corners += 1
        if ((ww - 1) == selectedCon[a][0][0] and (hh - 1) == selectedCon[a][0][1]):
            corners += 1
        if ((ww - 1) == selectedCon[a][0][0] and 0 == selectedCon[a][0][1]):
            corners += 1

    # This will check for different threshold values and obntain different parts of the lesion as contours
    # to obtain the best extracted lesion
    if (percentage < 1 or (percentage < 3 and distance_to_center > 15) or percentage > 90 or
            (percentage < 10 and distance_to_center > 100) or corners > 0):
        value = 165
        existingPercentage = 0
        existingStatus = False
        common = selectedCon
        while (value > 50):
            con, per = variableContour(img, value)
            value -= 3
            if (per > 0):
                M = cv2.moments(con)
                try:
                    center_X = int(M["m10"] / M["m00"])
                    center_Y = int(M["m01"] / M["m00"])
                    contour_center = (center_X, center_Y)
                    distance_to_center = (distance.euclidean(image_center, contour_center))
                except:
                    distance_to_center = 200
                if (per > existingPercentage and
                        per < 75 and (distance_to_center < 100 or per > 2)):
                    existingStatus = True
                    existingPercentage = per
                    existingCon = con

        if (not (existingStatus)):
            existingCon = common

        maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
        maskBlotch = cv2.drawContours(maskBlotch, [existingCon], 0, (255, 255, 255), cv2.FILLED)
        blotchArea = np.sum(maskBlotch != 0)
        blotchCon = existingCon
        percentage = blotchArea / (hh * ww) * 100

        M = cv2.moments(existingCon)
        center_X = int(M["m10"] / M["m00"])
        center_Y = int(M["m01"] / M["m00"])
        contour_center = (center_X, center_Y)
        distance_to_center = (distance.euclidean(image_center, contour_center))
        if ((percentage < 5 and distance_to_center > 100) or percentage > 75):
            return (img, blotchCon, 100, blotchCon, img)
        else:
            i = img
            result = cv2.bitwise_and(i, i, mask=maskBlotch)
            maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
            hull = cv2.convexHull(existingCon)
            maskBlotch = cv2.drawContours(maskBlotch, [hull], 0, (255, 255, 255), cv2.FILLED)
            hullResult = cv2.bitwise_and(i, i, mask=maskBlotch)
            return (result, blotchCon, percentage, hull, hullResult)
    return (result, selectedCon, percentage, hull, hullResult)


def variableContour(img, value):
    src = img
    # Getting the hight and the width of the image
    hh, ww = img.shape[:2]
    src = cv2.GaussianBlur(src, (5, 5), 0)
    src3 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    kernal = np.ones((7, 7), np.uint8)
    src2 = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernal)
    src1 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
    retOther, threshOther = cv2.threshold(src1, value, 255, cv2.THRESH_BINARY_INV)
    maskOther = cv2.cvtColor(threshOther, cv2.COLOR_BGR2RGB)
    im_thresh_color_other = cv2.bitwise_and(src3, maskOther)
    gray = cv2.cvtColor(im_thresh_color_other, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    i = len(cntsSorted) - 1
    if (len(cntsSorted) > 0):
        status = True
        valid = True
        while status:
            if (i < 0):
                status = False
                valid = False
                break
            ca = cntsSorted[i]
            count = 0
            for a in range(len(ca)):
                if ((ww - 5) < ca[a][0][0]):
                    count += 1
            for a in range(len(ca)):
                if ((hh - 5) < ca[a][0][1]):
                    count += 1
            if (0 in ca) or (count != 0):
                i -= 1
            else:
                status = False

        if (valid):
            maskBlotch = np.zeros((hh, ww), dtype=np.uint8)
            maskBlotch = cv2.drawContours(maskBlotch, [ca], 0, (255, 255, 255), cv2.FILLED)
            blotchArea = np.sum(maskBlotch != 0)
            blotchCon = ca
        else:
            blotchArea = 0
            blotchCon = None

    else:
        blotchArea = 0
        blotchCon = None

    percentage = blotchArea / (hh * ww) * 100

    return (blotchCon, percentage)


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


# Asymmetry, Border, Colour, Diameter, Globules, Blotches, RedAreas, Rosettes, RegressionStructure,
# BlueWhite, AtypicalNetwork, Streaks, Asymmetry_Real, Border_New, Colour_Real, Diameter_Real
def checkMelanoma(file):
    try:
        image = file

        image = removeLens(image)

        image = removeHair(image)

        image = removeInkPatches(image)

        copy = image
        image, con, percentage, hull, hullImage = getContour(image)

        arr = []
        arr.append([])

        if (percentage == 100):
            state = 0

            asymmetryValue = 0
            percentageValue = 0
            asymmetryRealValue = 0
            borderValue = 0
            borderRealValue = 0
            borderNewValue = 0
            try:
                brown, red, blue, white, black, result, count = detectColour(copy, con, state)
                colourValue = result
                colourRealValue = count
            except Exception as e:
                colourValue = "error"
                colourRealValue = "error"
                print("Colour", e)
            diameterValue = 0
            widthValue = 0
            heightValue = 0
            diameterRealValue = 0
            try:
                result = detectGlobules(copy, con)
                globulesValue = result
            except Exception as e:
                globulesValue = "error"
                print("Globules", e)

            try:
                result = detectBlotches(copy, con)
                blotchesValue = result
            except Exception as e:
                blotchesValue = "error"
                print("Blotches", e)

            try:
                result = detectMilkyRedAreas(copy, con)
                redValue = result
            except Exception as e:
                redValue = "error"
                print("RedAreas", e)

            try:
                result = detectRosettes(copy, con, state)
                rosettesValue = result
            except Exception as e:
                rosettesValue = "error"
                print("Rosettes", e)

            try:
                result = detectRegressionStructure(copy, con, state)
                regressionValue = result
            except Exception as e:
                regressionValue = "error"
                print("RegressionStructure", e)

            try:
                result = detectBlueWhiteVeil(copy, con, state)
                blueValue = result
            except Exception as e:
                blueValue = "error"
                print("BlueWhite", e)

            try:
                result = detectAtypicalNetwork(copy, con, state)
                atypicalValue = result
            except Exception as e:
                atypicalValue = "error"
                print("AtypicalNetwork", e)

            streaksValue = 0

            arr[0].append(asymmetryValue)
            arr[0].append(borderValue)
            arr[0].append(colourValue)
            arr[0].append(diameterValue)
            arr[0].append(globulesValue)
            arr[0].append(blotchesValue)
            arr[0].append(redValue)
            arr[0].append(rosettesValue)
            arr[0].append(regressionValue)
            arr[0].append(blueValue)
            arr[0].append(atypicalValue)
            arr[0].append(streaksValue)
            arr[0].append(asymmetryRealValue)
            arr[0].append(borderNewValue)
            arr[0].append(colourRealValue)
            arr[0].append(diameterRealValue)
        else:
            state = 1
            try:
                result, amount, perimeter = detectAsymmetry(hullImage, hull)
                asymmetryValue = result
                percentageValue = perimeter
                asymmetryRealValue = amount
            except Exception as e:
                asymmetryValue = "error"
                asymmetryRealValue = "error"
                print("Asymmetry", e)

            try:
                result, amount, amountNew = detectBorderIrregularity(con, hull)
                borderValue = result
                borderRealValue = amount
                borderNewValue = amountNew
            except Exception as e:
                borderValue = "error"
                borderRealValue = "error"
                borderNewValue = "error"
                print("Border", e)

            try:
                brown, red, blue, white, black, result, count = detectColour(hullImage, hull, state)
                colourValue = result
                colourRealValue = count
            except Exception as e:
                colourValue = "error"
                colourRealValue = "error"
                print("Colour", e)

            try:
                result, amount, width, height = detectDiameter(hullImage, hull)
                diameterValue = result
                widthValue = width
                heightValue = height
                diameterRealValue = amount
            except Exception as e:
                diameterValue = "error"
                widthValue = "error"
                heightValue = "error"
                diameterRealValue = "error"
                print("Diameter", e)

            try:
                result = detectGlobules(hullImage, hull)
                globulesValue = result
            except Exception as e:
                globulesValue = "error"
                print("Globules", e)

            try:
                result = detectBlotches(copy, hull)
                blotchesValue = result
            except Exception as e:
                blotchesValue = "error"
                print("Blotches", e)

            try:
                result = detectMilkyRedAreas(copy, hull)
                redValue = result
            except Exception as e:
                redValue = "error"
                print("RedAreas", e)

            try:
                result = detectRosettes(hullImage, hull, state)
                rosettesValue = result
            except Exception as e:
                rosettesValue = "error"
                print("Rosettes", e)

            try:
                result = detectRegressionStructure(hullImage, hull, state)
                regressionValue = result
            except Exception as e:
                regressionValue = "error"
                print("RegressionStructure", e)

            try:
                result = detectBlueWhiteVeil(hullImage, hull, state)
                blueValue = result
            except Exception as e:
                blueValue = "error"
                print("BlueWhite", e)

            try:
                result = detectAtypicalNetwork(hullImage, hull, state)
                atypicalValue = result
            except Exception as e:
                atypicalValue = "error"
                print("AtypicalNetwork", e)

            try:
                result = detectStreaks(hullImage, hull)
                streaksValue = result
            except Exception as e:
                streaksValue = "error"
                print("Streaks", e)

            arr[0].append(asymmetryValue)
            arr[0].append(borderValue)
            arr[0].append(colourValue)
            arr[0].append(diameterValue)
            arr[0].append(globulesValue)
            arr[0].append(blotchesValue)
            arr[0].append(redValue)
            arr[0].append(rosettesValue)
            arr[0].append(regressionValue)
            arr[0].append(blueValue)
            arr[0].append(atypicalValue)
            arr[0].append(streaksValue)
            arr[0].append(asymmetryRealValue)
            arr[0].append(borderNewValue)
            arr[0].append(colourRealValue)
            arr[0].append(diameterRealValue)

    except Exception as e:
        print("error")
        print("Main Loop", e)

    filename = 'model/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(arr)
    y_pred_proba = loaded_model.predict_proba(arr)[::, 1]
    status = ''
    level = round(y_pred_proba[0] * 100, 2)
    if (y_pred[0] == 1):
        status = 'Positive'
    else:
        status = 'Negative'
    return (status, level, arr)
