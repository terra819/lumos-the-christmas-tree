#trainer.py
#assume images are organized in the correct directories
#ex: /trainers/lumos/image.jpg
#.            /nox/image1.jpg
#names of directories and images do not matter, one directory per spell to yrain on

TRAINER_DIRECTORY = "Trainers" #directory containing sorted images
MIN_IMAGES_REQUIRED = 5 #for best results, do not set lower than 4

def createTrainer:
    globals _spellLabels, _cellWidth, _cellHeight, _imagesPerSpell, _trainerWidth, _trainerHeight
    #get relative path, append to path
    #get directory contents
    _spellLabels = []
    #get image sizes
    _cellWidth =
    _cellHeight =
    #if all images are not same size, stop. must be same size
    #determine min number of images in each dir
    _imagesPerSpell =
    if _imagesPerSpell < MIN_IMAGES_REQUIRED:
        #stop, need more images
    #calculate size of trainer image
    _trainerWidth = _imagesPerSpell * _cellWidth
    _trainerHeight = _spellDirectories.len * _cellHeight
    #create blank trainer image trainer.png
    #loop through directories, paste tmages to trainer
    _pasteAt = (0,0)
    #foreach spell in _spellLabels
        #foreach img in spell
           #paste img at _pasteAt
            _pasteAt = (_pasteAt[0] + _cellWidth, _pasteAt[1])
        _pasteAt = (_pasteAt[0], _pasteAt[1] + _cellHeight)
   
    #trainer image createdeated

createTrainer()
trainSpells()