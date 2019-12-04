import numpy as np
import cv2
import time
import math

DEQUE_BUFFER_SIZE = 40
TRACE_THICKNESS = 4
BGS_HISTORY_FRAMES = 200
ENABLE_SAVE_IMAGE = False
ENABLE_SPELL_TRAINING  = False
TRAINED_SPELL_MODEL = "spellsModel.yml"
TRAINER_IMAGE_WIN_SIZE = 64
GESTURE_TRAINER_IMAGE = "gesuretrainer.jpg"  
NO_OF_IMAGES_PER_ELEMENT = 20
fileNum = 0  
ESC_KEY = 27
SPACE_KEY = 32
ALT_KEY = 18
windowName = "Wand Trace Window"
MIN_0_TRACE_AREA = 7600    #for M
MIN_1_TRACE_AREA = 30000   #for 0
MIN_2_TRACE_AREA = 12500   #for '4'
MIN_3_TRACE_AREA = 23000   #for ~
CROPPED_IMG_MARGIN = 10      #pixels
MAX_TRACE_SPEED = 150     #pixels/second (30p/0.2sec)
deviceID = 0

camera = cv2.VideoCapture(deviceID)
_frameWidth = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
_frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
_wandMoveTracingFrame = np.zeros((_frameHeight,_frameWidth,1), np.uint8) # (that is: height, width,numchannels)
cameraFrame = np.zeros((_frameHeight,_frameWidth,1), np.uint8)
_pMOG2 = cv2.bgsegm.createBackgroundSubtractorMOG(BGS_HISTORY_FRAMES)

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

_blobDetector = cv2.SimpleBlobDetector_create(_params)

def get_hog() : 
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

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

# _hog = cv2.HOGDescriptor(
#     (64, 64), #winSize  50 x 50
#     (32, 32), #blocksize 32 x 32
#     (16, 16), #blockStride, 16 x 16
#     (16, 16), #cellSize, 16 x 16
#     9, #nbins,
#     1, #derivAper,
#     -1, #winSigma,
#     0, #histogramNormType,
#     0.2, #L2HysThresh,
#     0,#gammal correction,
#     64,#nlevels=64
#     1)

_tracePoints = []
_blobKeypoints = []
_lastKeypointTime = time.time()
    #return

def _wandDetect(frameData):
    global cameraFrame

    #Background Elimination
#     bgSubtractedFrame = _pMOG2.apply(frameData);
    cameraFrame = frameData         
    
    # Detect blobs
    keypoints = _blobDetector.detect(cameraFrame)

    # Show keypoints
    im_with_keypoints = cv2.drawKeypoints(cameraFrame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Debug Window", im_with_keypoints)
    
    return keypoints

def getWandTrace(frameData):
    global _lastKeypointTime, _tracePoints,_blobKeypoints
    _blobKeypoints = _wandDetect(frameData)
    
    #Add keypoints to deque. For now, take only the first found keypoint
    if(len(_blobKeypoints) > 0 ):
        currentKeypointTime = time.time()

        if (len(_tracePoints) > 0):
            elapsed = currentKeypointTime - _lastKeypointTime
#             print("elapsed")
#             print(elapsed)
            pt1 = (_tracePoints[len(_tracePoints) - 1])
            pt2 = (_blobKeypoints[0])
            distance = _distance(pt1,pt2)
#             print("distance")
#             print(distance)
            speed = distance / elapsed
#             print("speed")
#             print(speed)
            if (speed >= MAX_TRACE_SPEED):
#                 print("too fast")
#                 print(speed)
                return _wandMoveTracingFrame
            if (len(_tracePoints) >= DEQUE_BUFFER_SIZE):
                _tracePoints.pop(0)

            _tracePoints.append(_blobKeypoints[0])
            cv2.line(_wandMoveTracingFrame, (int(pt1.pt[0]), int(pt1.pt[1])), (int(pt2.pt[0]), int(pt2.pt[1])), 255, TRACE_THICKNESS)
        else:
            _lastKeypointTime = currentKeypointTime
            _tracePoints.append(_blobKeypoints[0])
    
    return _wandMoveTracingFrame

def _distance(pt1, pt2):
    return math.sqrt((pt1.pt[0] - pt2.pt[0])*(pt1.pt[0] - pt2.pt[0]) + (pt1.pt[1] - pt2.pt[1])*(pt1.pt[1] - pt2.pt[1]))

def wandVisible():
    global _blobKeypoints
    if(len(_blobKeypoints) == 0):
        return False
    return True

def checkTraceValidity():
    global _traceUpperCorner, _traceLowerCorner, _blobKeypoints,_tracePoints,_traceUpperCorner,_traceLowerCorner
    if (len(_blobKeypoints) == 0):    
        currentKeypointTime = time.time()
        elapsed = currentKeypointTime - _lastKeypointTime
        if (elapsed < 5.0):
            return False
        
        if (len(_tracePoints) > DEQUE_BUFFER_SIZE - 5):        
            _traceUpperCorner = (_frameWidth, _frameHeight)
            _traceLowerCorner = (0, 0)

            #Draw a trace by connecting all the keypoints stored in the deque
            #Also update lower and upper bounds of the trace
            for i in range(len(_tracePoints)):           
                if (_tracePoints[i].size == -99.0):
                    continue
                pt1 = (_tracePoints[i - 1].pt[0], _tracePoints[i - 1].pt[1])
                pt2 = (_tracePoints[i].pt[0], _tracePoints[i].pt[1])

                #Min x,y = traceUpperCorner points
                #Max x,y = traceLowerCorner points
                
                if (pt1[0] < _traceUpperCorner[0]):
                    _traceUpperCorner = (pt1[0], _traceUpperCorner[1])
                if (pt1[0] > _traceLowerCorner[0]):
                    _traceLowerCorner = (pt1[0], _traceLowerCorner[1])
                if (pt1[1] < _traceUpperCorner[1]):
                    _traceUpperCorner = (_traceUpperCorner[0], pt1[1])
                if (pt1[1] > _traceLowerCorner[1]):
                    _traceLowerCorner = (_traceLowerCorner[0], pt1[1])
                    
            traceArea = (_traceLowerCorner[0] - _traceUpperCorner[0]) * (_traceLowerCorner[1] - _traceUpperCorner[1])
            
            if (traceArea > MIN_0_TRACE_AREA):
                return True
        
        #It's been over five seconds since the last keypoint and trace isn't valid
        eraseTrace()
    
    return False

def eraseTrace():
    global _wandMoveTracingFrame, _tracePoints
    #Erase existing trace
    _wandMoveTracingFrame = np.zeros((_frameHeight,_frameWidth,1), np.uint8)
    #Empty corresponding tracePoints
    _tracePoints = []
        
def _cropSaveTrace():
    global _traceUpperCorner, _traceLowerCorner
    if (_traceUpperCorner[0] > CROPPED_IMG_MARGIN):
        _traceUpperCorner = (_traceUpperCorner[0] - CROPPED_IMG_MARGIN, _traceUpperCorner[1])
    else:
        _traceUpperCorner = (0, _traceUpperCorner[1])

    if (_traceUpperCorner[1] > CROPPED_IMG_MARGIN):
        _traceUpperCorner = (_traceUpperCorner[0], _traceUpperCorner[1] - CROPPED_IMG_MARGIN)
    else:
        _traceUpperCorner = (_traceUpperCorner[0], 0)

    if (_traceLowerCorner[0] < _frameWidth - CROPPED_IMG_MARGIN):
        _traceLowerCorner = (_traceLowerCorner[0] + CROPPED_IMG_MARGIN, _traceLowerCorner[1])
    else:
        _traceLowerCorner = (_frameWidth, _traceLowerCorner[1])
        
    if (_traceLowerCorner[1] < _frameHeight - CROPPED_IMG_MARGIN):
        _traceLowerCorner = (_traceLowerCorner[0], _traceLowerCorner[1] + CROPPED_IMG_MARGIN)
    else:
        _traceLowerCorner = (_traceLowerCorner[0], _frameHeight)


    traceWidth = int(_traceLowerCorner[0] - _traceUpperCorner[0])
    traceHeight = int(_traceLowerCorner[1] - _traceUpperCorner[1])

    if (traceHeight > traceWidth):
        _sizeheight = int(TRAINER_IMAGE_WIN_SIZE)
        _sizewidth = int(traceWidth * TRAINER_IMAGE_WIN_SIZE / traceHeight)  #Since traceHeight & traceWidth are always gonna be > TRAINER_IMAGE_WIN_SIZE
     
    else:
        _sizewidth = int(TRAINER_IMAGE_WIN_SIZE)
        _sizeheight = int(traceHeight * TRAINER_IMAGE_WIN_SIZE / traceWidth)

    clone = _wandMoveTracingFrame.copy()
    crop = clone[int(_traceUpperCorner[1]):int(_traceLowerCorner[1]), int(_traceUpperCorner[0]):int(_traceLowerCorner[0])]
    resizedCroppedTrace = cv2.resize(crop, (_sizewidth, _sizeheight))
    _finalTraceCell = np.zeros((TRAINER_IMAGE_WIN_SIZE, TRAINER_IMAGE_WIN_SIZE,1), np.uint8)
    for i in range(resizedCroppedTrace.shape[0]):
        for j in range(resizedCroppedTrace.shape[1]):
            _finalTraceCell[(i,j)] = resizedCroppedTrace[(i,j)]
#     cv2.imshow("_finalTraceCell window", _finalTraceCell)
#     cv2.imwrite("spellTraceCell.png", _finalTraceCell)
    return _finalTraceCell

def recognizeSpell():
    print("recognizeSpell")
    finalTrace = _cropSaveTrace()
    cv2.imwrite("finalTrace.png", finalTrace)
#     finalTrace = cv2.imread("Untitled.png", 0)
#     print(finalTrace.shape)
#     print(finalTrace.dtype)
#     print("finalTrace")
    deskewedTrace = _deskew(finalTrace)
    print("deskewedTrace")
    cv2.imwrite("deskewedTrace.png", deskewedTrace)
#     cv2.imshow("finalTrace window", finalTrace)
#     cv2.imshow("deskewedTrace window", deskewedTrace)
#     cv2.imshow("finalTrace window", finalTrace)
#     time.sleep(50)
#     print("deskewedTrace")
#     print(deskewedTrace)
#     hog = cv2.HOGDescriptor()
    hog = get_hog()
    print("get_hog")
    descriptors = hog.compute(deskewedTrace)
    print("hog.compute(deskewedTrace)")
#     descriptors = np.float32(descriptors)
    descriptors = np.squeeze(descriptors)
    print("descriptors")
#     print(descriptors.dtype)
#     print(descriptors.shape)
#     cv2.imwrite("descriptors.png", descriptors)
# #     print("descriptors")
#     descriptorMatrix = _ConvertVectortoMatrix(descriptors)
#     print("descriptorMatrix")

#     print(testimg.shape)
#     #convert 2d to 1d 
#     testMat = testimg.copy().reshape(-1)
#     print(testMat.shape)
#     testMat.convertTo(testMat, CV_32F)
#     res = np.float32(testMat)
#     print("_ConvertVectortoMatrix")
#     print(descriptorMatrix.shape)
    svm = cv2.ml.SVM_load(TRAINED_SPELL_MODEL)
    svm.setGamma(0.50625)
    svm.setC(12.5)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    print("svm")
#     image = np.float32(cv2.imread("spellTraceCell.png", 0))
#     print(svm.predict(np.ravel(image)[None, :]))
#     print("svm.var_count")
#     print(svm.var_count)
#     testImg = np.float(cv2.imread("spellTraceCell.png",0))
#     print(svm.predict(np.ravel(finalTrace)))
#     print(svm.predict(np.ravel(finalTrace)[None, :]))
#     cv2.imwrite("finalTrace" + str(time.time()) + ".png", finalTrace)
#     cv2.imwrite("descriptorMatrix" + str(time.time()) + ".png", res)
#     print(res.shape)
#     print("descriptors")
#     print(descriptors)
#     print("descriptors.shape[1]")
#     print(descriptors.shape[1])
#     print("svm.getVarCount()")
#     print(svm.getVarCount())
#     cv2.imwrite("descriptors.png", descriptors)
    prediction = svm.predict(descriptors[None,:])[1].ravel()
    print("prediction")
    #cv2.error: OpenCV(4.1.1) /home/pi/opencv-python/opencv/modules/ml/src/svm.cpp:2011: error: (-215:Assertion failed) samples.cols == var_count && samples.type() == CV_32F in function 'predict'

    print(prediction)
    return prediction

def _ConvertVectortoMatrix(inHOG):
    descriptorMatrix = np.zeros((1, inHOG.shape[1]),np.float32)
    for i in range(inHOG.shape[0]):
        for j in range(inHOG.shape[1]):
            descriptorMatrix[(i,j)] = inHOG[(i,j)]
        
def spellRecognitionTrainer():
    trainCells = []
    trainLabels = []
    _loadTrainLabel(GESTURE_TRAINER_IMAGE, trainCells, trainLabels)
    print("trainCells")
    print(trainCells)
    print("trainLabels")
    print(trainLabels)
    deskewedTrainCells = []
    _CreateDeskewedTrain(deskewedTrainCells, trainCells)
    print("deskewedTrainCells")
    print(deskewedTrainCells)
    print("trainCells")
    print(trainCells)

    trainHOG =[]
    testHOG = []
    _CreateTrainHOG(trainHOG, deskewedTrainCells)
    print("trainHOG")
    print(trainHOG)
    print("deskewedTrainCells")
    print(deskewedTrainCells)

    descriptor_size = len(trainHOG[0])
    print("descriptor_size")
    print(descriptor_size)

    trainMat(len(trainHOG), descriptor_size, CV_32FC1)
    print("trainMat")
    print(trainMat)
    
    _ConvertVectortoMatrix(trainHOG, trainMat)
    print("_ConvertVectortoMatrix")
    print(_ConvertVectortoMatrix)

    _SVMtrain(trainMat, trainLabels)
    print("_SVMtrain")
    print(_SVMtrain)
                    
def _loadTrainLabel(pathName, trainCells, trainLabels):
    img = cv2.imread(pathName, 0)
    cv2.imwrite("img.png", imgopencv)
    ImgCount = 0
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            digitImg = (img.shape[1](j, j + TRAINER_IMAGE_WIN_SIZE).shape[0](i, i + TRAINER_IMAGE_WIN_SIZE)).copy()
            trainCells.append(digitImg)
            ImgCount = ImgCount + 1
            j = j + TRAINER_IMAGE_WIN_SIZE
        i = i + TRAINER_IMAGE_WIN_SIZE          

    print("Image Count : ")
    print(ImgCount)
    digitClassNumber = 0

    for Z in range(ImgCount):
            if (z % NO_OF_IMAGES_PER_ELEMENT == 0 and z != 0):
                    digitClassNumber = digitClassNumber + 1                    
            trainLabels.append(digitClassNumber)
            
def _CreateDeskewedTrain(deskewedTrainCells, trainCells):
    for i in range(len(trainCells)):
        deskewedImg = _deskew(trainCells[i])
        deskewedTrainCells.append(deskewedImg)

def _CreateTrainHOG(trainHOG, deskewedtrainCells):
    for y in range(len(deskewedtrainCells)):
        descriptors = _hog.compute(deskewedtrainCells[y])
        trainHOG.append(descriptors)

def _getSVMParams(svm):
    print("Kernel type     : ")
    print(svm.getKernelType())
    print("Type            : ")
    print(svm.getType())
    print("C               : ")
    print(svm.getC())
    print("Degree          : ")
    print(svm.getDegree())
    print("Nu              : ")
    print(svm.getNu())
    print("Gamma           : ")
    print(svm.getGamma())

def _SVMtrain(trainMat, trainLabels):
    svm = SVM.create()
    svm.setGamma(0.50625)
    svm.setC(12.5)
    svm.setKernel(SVM.RBF)
    svm.setType(SVM.C_SVC)
    td = TrainData.create(trainMat, ROW_SAMPLE, trainLabels)
    svm.train(td)
    svm.save("model4[1]ml")
    _getSVMParams(svm)
        
def _deskew(img):
    SZ = 64
#     affineFlags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


while(True):
    # Capture frame-by-frame
    retval, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #get wand trace frame
    wandTraceFrame = getWandTrace(gray)
    cv2.imshow(windowName, wandTraceFrame)
    
    if not ENABLE_SPELL_TRAINING and not ENABLE_SAVE_IMAGE:
        if (checkTraceValidity()):
            print("Trace valid for spell rcognition")
            spell = recognizeSpell()
            print(spell)
            
            if spell== 0:
                print("*** 0: Music Toggle ***")
            elif spell == 1:
                print("*** 1: Blinds Toggle ***")
            elif spell == 2:
                print("*** 2: Bot Move ***")
            elif spell == 3:
                print("*** 3: Lights Toggle ***")
            else:
                print("That's not a spell")
            
            eraseTrace()
       
    waitKey = cv2.waitKey(10)
    if waitKey == ord('y'):
        print("spelltraining")
        if ENABLE_SPELL_TRAINING:
            spellRecognitionTrainer()
        
    if waitKey == ord('s'):
        print("savemage")
        if ENABLE_SAVE_IMAGE:
            fileName = "Image" + str(fileNum) + ".png"
            cv2.imwrite(fileName, wandTraceFrame)
            fileNum = fileNum + 1
            eraseTrace()
            
    if waitKey == ord('c'):
        print("clear image")
        if ENABLE_SAVE_IMAGE:
            eraseTrace()
        
    if waitKey == ord('q'):
        print("quit")
        break

camera.release()
cv2.destroyAllWindows()

