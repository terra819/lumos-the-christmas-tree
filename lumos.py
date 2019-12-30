#Inspired by Sanni-t's techMagic app: 
#https://github.com/sanni-t/techMagicApp

import numpy as np
import cv2
import time
import math
import os
import spellcaster
import random

# Settings
DEQUE_BUFFER_SIZE = 40
TRACE_THICKNESS = 4
ENABLE_SAVE_IMAGE = False
TRAINER_IMAGE_WIN_SIZE = 64
windowName = "Wand Trace Window"
CROPPED_IMG_MARGIN = 10  # pixels
MAX_TRACE_SPEED = 150  # pixels/second (30p/0.2sec)
deviceID = 0

# Globals
camera = cv2.VideoCapture(deviceID)
frameWidth = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
wandMoveTracingFrame = np.zeros((frameHeight, frameWidth, 1), np.uint8)
cameraFrame = np.zeros((frameHeight, frameWidth, 1), np.uint8)
tracePoints = []
blobKeypoints = []
lastKeypointTime = time.time()
dir_path = os.path.dirname(os.path.realpath(__file__))
targetSamplesDirectory = dir_path + "/samples"
TRAINED_SPELL_MODEL = dir_path + "/spell_model.dat"
breakIn = random.randint(5,15)


def get_blob_detector():
    _params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    _params.minThreshold = 150
    _params.maxThreshold = 255

    # Fliter by color
    _params.filterByColor = True
    _params.blobColor = 255

    # Filter by Area.
    _params.filterByArea = True
    _params.minArea = 10
    _params.maxArea = 40

    # Filter by Circularity
    _params.filterByCircularity = True
    _params.minCircularity = 0.5

    # Filter by Convexity
    _params.filterByConvexity = True
    _params.minConvexity = 0.5

    # Filter by Inertia
    _params.filterByInertia = False

    return cv2.SimpleBlobDetector_create(_params)


def get_hog():
    winSize = (64, 64)
    blockSize = (32, 32)
    blockStride = (16, 16)
    cellSize = (16, 16)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                            winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    return hog


def _wandDetect(frameData):
    global cameraFrame
    cameraFrame = frameData

    # Detect blobs
    keypoints = _blobDetector.detect(cameraFrame)

    # Show keypoints
    im_with_keypoints = cv2.drawKeypoints(cameraFrame, keypoints, np.array(
        []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Debug Window", im_with_keypoints)

    return keypoints


def getWandTrace(frameData):
    global lastKeypointTime, tracePoints, blobKeypoints
    blobKeypoints = _wandDetect(frameData)

    # Add keypoints to deque. For now, take only the first found keypoint
    if(len(blobKeypoints) > 0):
        currentKeypointTime = time.time()

        if (len(tracePoints) > 0):
            elapsed = currentKeypointTime - lastKeypointTime
            pt1 = (tracePoints[len(tracePoints) - 1])
            pt2 = (blobKeypoints[0])
            distance = _distance(pt1, pt2)
            speed = distance / elapsed
            if (speed >= MAX_TRACE_SPEED):
                return wandMoveTracingFrame

            tracePoints.append(blobKeypoints[0])
            cv2.line(wandMoveTracingFrame, (int(pt1.pt[0]), int(pt1.pt[1])), (int(
                pt2.pt[0]), int(pt2.pt[1])), 255, TRACE_THICKNESS)
        else:
            lastKeypointTime = currentKeypointTime
            tracePoints.append(blobKeypoints[0])

    return wandMoveTracingFrame


def _distance(pt1, pt2):
    return math.sqrt((pt1.pt[0] - pt2.pt[0])*(pt1.pt[0] - pt2.pt[0]) + (pt1.pt[1] - pt2.pt[1])*(pt1.pt[1] - pt2.pt[1]))


def wandVisible():
    global blobKeypoints
    if(len(blobKeypoints) == 0):
        return False
    return True

def sizeTrace():
    global _traceUpperCorner, _traceLowerCorner, tracePoints
    _traceUpperCorner = (frameWidth, frameHeight)
    _traceLowerCorner = (0, 0)

    for i in range(len(tracePoints)):
        if (tracePoints[i].size == -99.0):
            continue
        pt1 = (tracePoints[i - 1].pt[0], tracePoints[i - 1].pt[1])
        if (pt1[0] < _traceUpperCorner[0]):
            _traceUpperCorner = (pt1[0], _traceUpperCorner[1])
        if (pt1[0] > _traceLowerCorner[0]):
            _traceLowerCorner = (pt1[0], _traceLowerCorner[1])
        if (pt1[1] < _traceUpperCorner[1]):
            _traceUpperCorner = (_traceUpperCorner[0], pt1[1])
        if (pt1[1] > _traceLowerCorner[1]):
            _traceLowerCorner = (_traceLowerCorner[0], pt1[1])

def checkTraceValidity():
    global _traceUpperCorner, _traceLowerCorner, tracePoints
    if (len(blobKeypoints) == 0):
        currentKeypointTime = time.time()
        elapsed = currentKeypointTime - lastKeypointTime
        if (elapsed < 4.0):
            return False

        if (len(tracePoints) > DEQUE_BUFFER_SIZE - 5):
            sizeTrace()
            return True
        eraseTrace()

    return False

def eraseTrace():
    global wandMoveTracingFrame, tracePoints
    wandMoveTracingFrame = np.zeros((frameHeight, frameWidth, 1), np.uint8)
    tracePoints = []

def _cropSaveTrace():
    global _traceUpperCorner, _traceLowerCorner
    if (_traceUpperCorner[0] > CROPPED_IMG_MARGIN):
        _traceUpperCorner = (
            _traceUpperCorner[0] - CROPPED_IMG_MARGIN, _traceUpperCorner[1])
    else:
        _traceUpperCorner = (0, _traceUpperCorner[1])

    if (_traceUpperCorner[1] > CROPPED_IMG_MARGIN):
        _traceUpperCorner = (
            _traceUpperCorner[0], _traceUpperCorner[1] - CROPPED_IMG_MARGIN)
    else:
        _traceUpperCorner = (_traceUpperCorner[0], 0)

    if (_traceLowerCorner[0] < frameWidth - CROPPED_IMG_MARGIN):
        _traceLowerCorner = (
            _traceLowerCorner[0] + CROPPED_IMG_MARGIN, _traceLowerCorner[1])
    else:
        _traceLowerCorner = (frameWidth, _traceLowerCorner[1])

    if (_traceLowerCorner[1] < frameHeight - CROPPED_IMG_MARGIN):
        _traceLowerCorner = (
            _traceLowerCorner[0], _traceLowerCorner[1] + CROPPED_IMG_MARGIN)
    else:
        _traceLowerCorner = (_traceLowerCorner[0], frameHeight)

    traceWidth = int(_traceLowerCorner[0] - _traceUpperCorner[0])
    traceHeight = int(_traceLowerCorner[1] - _traceUpperCorner[1])

    if (traceHeight > traceWidth):
        _sizeheight = int(TRAINER_IMAGE_WIN_SIZE)
        _sizewidth = int(traceWidth * TRAINER_IMAGE_WIN_SIZE / traceHeight)

    else:
        _sizewidth = int(TRAINER_IMAGE_WIN_SIZE)
        _sizeheight = int(traceHeight * TRAINER_IMAGE_WIN_SIZE / traceWidth)

    clone = wandMoveTracingFrame.copy()
    crop = clone[int(_traceUpperCorner[1]):int(_traceLowerCorner[1]), int(
        _traceUpperCorner[0]):int(_traceLowerCorner[0])]
    resizedCroppedTrace = cv2.resize(crop, (_sizewidth, _sizeheight))
    _finalTraceCell = np.zeros((TRAINER_IMAGE_WIN_SIZE, TRAINER_IMAGE_WIN_SIZE, 1), np.uint8)
    for i in range(resizedCroppedTrace.shape[0]):
        for j in range(resizedCroppedTrace.shape[1]):
            _finalTraceCell[(i, j)] = resizedCroppedTrace[(i, j)]
    return _finalTraceCell


def recognizeSpell():
    finalTrace = _cropSaveTrace()
    if ENABLE_SAVE_IMAGE:
        saveSample(finalTrace)
    deskewedTrace = _deskew(finalTrace)
    hog = get_hog()
    descriptors = hog.compute(deskewedTrace)
    descriptors = np.squeeze(descriptors)
    svm = cv2.ml.SVM_load(TRAINED_SPELL_MODEL)
    svm.setGamma(0.50625)
    svm.setC(12.5)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    prediction = svm.predict(descriptors[None, :])[1].ravel()
    print(prediction)
    return prediction

def saveSample(finalTrace):
    if not os.path.exists(targetSamplesDirectory):
        os.makedirs(targetSamplesDirectory)
    fileName = targetSamplesDirectory + "/Image" + str(time.time()) + ".png"
    cv2.imwrite(fileName, finalTrace)
    print("Image saved as " + fileName)

def _deskew(img):
    SZ = 64
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(
        img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

_blobDetector = get_blob_detector()
spellCount = 0
broken = False

while(True):
    # Capture frame-by-frame
    retval, frame = camera.read()
    flipped = cv2.flip(frame, 1)
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)

    # get wand trace frame
    wandTraceFrame = getWandTrace(gray)
    cv2.imshow(windowName, wandTraceFrame)

    if (checkTraceValidity()):
        spell = recognizeSpell()

        # Todo: Get probability % and set min value
        text = "Unknown"
        if(spellCount != breakIn):
            if not broken:
                if spell == 0:
                    text = "Arresto Momentum"
                    spellcaster.arresto_momentum()
                elif spell == 1:
                    text = "Finite Incantatem"
                    spellcaster.finite_incantatem()
                elif spell == 3:
                    text = "Incendio"
                    spellcaster.incendio()
                elif spell == 4:
                    text = "Nox"
                    spellcaster.nox()
                elif spell == 5:
                    text = "Lumos"
                    spellcaster.lumos()
                elif spell == 6:
                    text = "Aguamenti"
                    spellcaster.aguamenti()
                elif spell == 7:
                    text = "Silencio"
                    spellcaster.silencio()
            else: 
                #currently broke! can only perform reparo spell:
                if spell == 2:
                    text = "Reparo"
                    broken = False

                    # reset so it can break again
                    spellCount = 0
                    breakIn = random.randint(5,15)
                    spellcaster.reparo()
        else:
            # randomly 'break'
            text = "Broken!"
            broken = True
            spellcaster.broken()

        # Show the user what was detected
        wandTraceFrame = cv2.putText(wandTraceFrame, text, (00, 185), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA, False) 
        cv2.imshow(windowName, wandTraceFrame)
        cv2.waitKey(3000)    

        spellCount = spellCount + 1
        eraseTrace()

    waitKey = cv2.waitKey(10)

    if waitKey == ord('q'):
        print("Quitting")
        break

# stop all spells
spellcaster.finite_incantatem()
camera.release()
cv2.destroyAllWindows()
