#trainer.py
# Assumes images are sorted in the correct directories
# Ex: /training_images/lumos/image.jpg
#                    /nox/image1.jpg
# Names of directories are not important
# Names of images inside aren't important
import cv2
import numpy as np
import os
from common import mosaic

TRAINER_DIRECTORY = "training_images" #directory containing sorted images
MIN_IMAGES_REQUIRED = 4 #for best results, do not set lower than 5

dir_path = os.path.dirname(os.path.realpath(__file__))
targetTrainersDirectory = dir_path + "/" + TRAINER_DIRECTORY
gestureTrainerImg = dir_path + "/training_image.png"
modelOutput = dir_path + "/spell_model.dat"

def createTrainingImage():
    global _spellLabels, _cellWidth, _cellHeight, _imagesPerSpell

    if not os.path.exists(targetTrainersDirectory):
        print("can't find trainer directory")
        # directory not found, stop
        return

    #get image sizes
    _cellWidth = 0
    _cellHeight = 0
    _validSize = True   
    _imagesPerSpell = 0
    
    for (root,dirs,files) in os.walk(targetTrainersDirectory, topdown=True): 
        _spellCount = 0
        if len(dirs) > 0:
            _spellLabels = dirs
        for file in files:
            if ".png" in file:
                _spellCount = _spellCount + 1
                path = os.path.join(root, file)
                img = cv2.imread(path, 0)
                h = img.shape[0]
                w = img.shape[1]
                if (_cellWidth == 0 and _cellHeight == 0):
                    _cellWidth = w
                    _cellHeight = h
                elif (_cellWidth != w or _cellHeight != h):
                    # image is not the same dimensions as previously calculated images
                    # can't continue, all images must be same size
                    _validSize = False
                    break
        #if all images are not same size, stop. must be same size
        if not _validSize: 
            print("Images must all be the same size. Images detected that are different sizes")
            return
        if _spellCount > 0:
            if _imagesPerSpell == 0:
                _imagesPerSpell = _spellCount
            elif _spellCount < _imagesPerSpell:
                _imagesPerSpell = _spellCount
            #determine min number of images in each dir
            if (_imagesPerSpell < MIN_IMAGES_REQUIRED):
                #stop, need more images
                print("Not enough images for each spell. Each directory must contain at least " + str(MIN_IMAGES_REQUIRED) + " images")
                return

    _imgRows = []
    for (root,dirs,files) in os.walk(targetTrainersDirectory, topdown=True): 
        _imgRow = []
        for file in files:
            if len(_imgRow) < _imagesPerSpell:
                if ".png" in file:
                    path = os.path.join(root, file)
                    img = cv2.imread(path, 0)
                    _imgRow.append(img)
        _spellRow = cv2.hconcat(_imgRow)
        if _spellRow is not None:
            _imgRows.append(_spellRow)
    _matrix = cv2.vconcat(_imgRows)
    cv2.imwrite(gestureTrainerImg, _matrix)

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_digits(fn):
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (_cellWidth, _cellHeight))
    labels = np.repeat(np.arange(len(_spellLabels)), len(digits)/len(_spellLabels)) #todo replace with _spellLbels
    return digits, labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*_cellWidth*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (_cellWidth, _cellHeight), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):

        return self.model.predict(samples)[1].ravel()

def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err)*100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        
        vis.append(img)
    return mosaic(25, vis)

def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, _cellWidth*_cellHeight) / 255.0

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

# adapted from https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
def trainSpells():
    print("images per spell: " + str(_imagesPerSpell))
    print("number of spells: " + str(len(_spellLabels)))
    print('Loading digits from ' + gestureTrainerImg + ' ... ')
    # Load data.
    digits, labels = load_digits(gestureTrainerImg)

    print('Shuffle data ... ')
    # Shuffle data
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    print('Deskew images ... ')
    digits_deskewed = list(map(deskew, digits))

    print('Defining HoG parameters ...')
    # HoG feature descriptor
    hog = get_hog()

    print('Calculating HoG descriptor for every image ... ')
    hog_descriptors = []
    for img in digits_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    print('Spliting data into training (90%) and test set (10%)... ')
    train_n=int(0.9*len(hog_descriptors))
    digits_train, digits_test = np.split(digits_deskewed, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])


    print('Training SVM model ...')
    model = SVM()
    model.train(hog_descriptors_train, labels_train)

    print('Saving SVM model ...')
    model.save(modelOutput)


    # print('Evaluating model ... ')
    vis = evaluate_model(model, digits_test, hog_descriptors_test, labels_test)
    # cv2.imwrite("digits-classification.jpg",vis)
    # cv2.imshow("Vis", vis)
    # cv2.waitKey(0)
    
createTrainingImage()
trainSpells()