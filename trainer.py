#trainer.py
# Assumes images are sorted in the correct directories
# Ex: /training_images/lumos/image.jpg
#                    /nox/image1.jpg
# Names of directories will be used as spell labels
# Names of images inside aren't important
import cv2
import numpy as np
import os

TRAINER_DIRECTORY = "training_images" #directory containing sorted images
MIN_IMAGES_REQUIRED = 5 #for best results, do not set lower than 5

dir_path = os.path.dirname(os.path.realpath(__file__))
targetTrainersDirectory = dir_path + "/" + TRAINER_DIRECTORY
print(targetTrainersDirectory)

def createTrainingImage():
    global _spellLabels, _cellWidth, _cellHeight, _imagesPerSpell, _trainerWidth, _trainerHeight

    if not os.path.exists(targetTrainersDirectory):
        print("can't find trainer directory")
        # directory not found, stop
        return
    
    #get directory contents
    # print([name for name in os.listdir(".") if os.path.isdir(targetTrainersDirectory)])
    # _spellLabels = [name for name in os.listdir(".") if os.path.isdir(targetTrainersDirectory)]

    #get image sizes
    _cellWidth = 0
    _cellHeight = 0
    _validSize = True   
    _minSpellCount = 0 
    _imgRows = []
    _spellLabels = []
    # for i in range(len(_spellLabels)):

        # print("checking contents of " + targetTrainersDirectory + _spellLabels[i])
        # r=root, d=directories, f = files
    for r, d, f in os.walk(targetTrainersDirectory):
        _spellCount = 0
        _imgRow = []
        _spellLabels.append(d)
        for file in f:
            if ".png" in file:
                _spellCount = _spellCount + 1
                path = os.path.join(r, file)
                print("found file: " + path)
                img = cv2.imread(path, 0)
                h = img.shape[0]
                w = img.shape[1]
                print('width: ', w)
                print('height:', h)
                if (_cellWidth == 0 and _cellHeight == 0):
                    _cellWidth = w
                    _cellHeight = h
                elif (_cellWidth != w or _cellHeight != h):
                    # image is not the same dimensions as previously calculated images
                    # can't continue, all images must be same size
                    _validSize = False
                    break
                _imgRow.append(img)
        #if all images are not same size, stop. must be same size
        if not _validSize: 
            print("Images must all be the same size. Images detected that are different sizes")
            return
        if _spellCount > 0:
            if _minSpellCount == 0:
                _minSpellCount = _spellCount
            elif _spellCount < _minSpellCount:
                _minSpellCount = _spellCount
        #determine min number of images in each dir
        if _minSpellCount < MIN_IMAGES_REQUIRED:
            #stop, need more images
            print(_spellLabels)
            print("Not enough images for each spell. Each directory must contain at least " + str(MIN_IMAGES_REQUIRED) + " images")
            return
        _spellRow = cv2.hconcat(_imgRow)
        _imgRows.append(_spellRow)
    _matrix = cv2.vconcat(_imgRows)
    fileName = dir_path + "/Trainer.png"
    cv2.imwrite(fileName, _matrix)
        
createTrainingImage()
# trainSpells()