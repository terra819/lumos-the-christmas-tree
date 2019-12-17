# lumos-the-christmas-tree
Lumos the Christmas Tree with your Harry Potter Wand. Raspberry Pi, Neopixels, Infrared Night Vision IR Camera for Raspberry Pi, OpenCv, FadeCandy

## Introduction
This project is for my children. I wanted to give them a little piece of Hogwarts and teach them about programming. 

Running `lumos.py` will bring up 2 camera windows. 1 window is for debugging purposes, so you can see what is happening, and can tell if you are within the frame, and if there are any errant lights messing with your wand tracing. 

The 2nd window is the output of the wand trace. The program is configured to look for the small bright IR light reflected from the tip of the Harry Potter wands and captured by the night vision pi camera. The program will trace the wand light as best it can (its not perfect!), it works best in low light conditions. 

After a trace is completed, the program uses opencv to predict the correct spell based on the models trained in the `spell_model.dat` file. 

Then, once a spell has been interpreted, its corresponding function in `spellcaster.py` is triggered. Some spells trigger lights and music, some do other things. You may mix it up as you wish. 

*Note: I do not own the rights to the mp3 files, so you will need to put your own files and filenames to play them in `spellcaster.py`

## Hardware
You may configure this to work with other setups, but here is what I've tested with: 
1. [I-VOM Wireless Mini Speaker with 3.5mm Aux Input Jack, 3W Loud Portable Speaker for iPhone iPod iPad Cellphone Tablet Laptop, with USB Rechargeable Ba](https://www.amazon.com/gp/product/B07KQ44VGQ/ref=ppx_yo_dt_b_asin_title_o01_s00?ie=UTF8&psc=1)

2. [AmazonBasics USB 2.0 Cable - A-Male to Mini-B Cord - 6 Feet (1.8 Meters](https://www.amazon.com/gp/product/B00NH11N5A/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1)

3. [Infrared Night Vision IR Camera for Raspberry Pi 4, Pi 3b+ Video Webcam with Case Suits for 3D Priter](https://www.amazon.com/gp/product/B07T22X3PQ/ref=ppx_yo_dt_b_asin_title_o04_s00?ie=UTF8&psc=1)

4. [
Adafruit FadeCandy - Dithering USB-Controlled Driver for RGB NeoPixels [ADA1689]](https://www.amazon.com/gp/product/B00K9M3VLE/ref=ppx_yo_dt_b_asin_title_o04_s01?ie=UTF8&psc=1)

5. [
ALITOVE 50pcs DC 12V WS2811 Led Pixel Black 12mm Diffused Digital RGB Addressable Dream Color Round LED Pixels Module IP68 Waterproof](https://www.amazon.com/gp/product/B06XN66ZY6/ref=ppx_yo_dt_b_asin_title_o09_s00?ie=UTF8&psc=1)

6. [CanaKit Raspberry Pi 4 4GB Starter Kit - 4GB RAM](https://www.amazon.com/gp/product/B07V5JTMV9/ref=ppx_yo_dt_b_asin_title_o04_s00?ie=UTF8&psc=1)

7. Harry Potter wand from Universal Studios (or [make your own](https://www.hackster.io/news/build-your-own-magical-harry-potter-wand-for-far-less-than-55-e096a082579f))

## Software

1. [OpenCV 4](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/)

2. [fadecandy](https://github.com/scanlime/fadecandy)

## Training Images
This project comes with pre-programmed "spells" but you can train your own by using the steps below: 

1. Edit `lumos.py` and set `ENABLE_SAVE_IMAGE` to `True`
2. Then run `lumos.py` and draw your spells with your harry potter wand. The program will run as normal except it will save each spell traced. After tracing your images (at least 10 per spell for best results), stop the program (q) and examine your samples folder. The samples folder will contain all of your traced spells. 
3. Create a folder called `training_images` in this project
4. Create folders wthin the `training_images`;  1 folder for each spell. The names of the folders inside `training_images` does not matter to the program. 
5. Sort your images from the sample directory into your separate folders inside `training_images`
ex: 
```
training_images
|___aguamenti
|   |   image1.jpg
|   |   image2.jpg
|___incendio
    |   image4.jpg
    |   image8.jpg
```

Each folder inside `training_images` indicates a spell that the program will train on. 

6. After finished sorting sample images into their directories, run `trainer.py`. it will output a `training_image.png` and `spell_model.dat` file

7. In the lumos.py file, you will need to match up your new spells with the indexes of the spell model. You can see how this is already done. ex: 
``` py
if spell == 0:
    text = "Arresto Momentum"
    spellcaster.arresto_momentum()
elif spell == 1:
    text = "Finite Incantatem"
    spellcaster.finite_incantatem()
elif spell == 2:
    text = "Reparo"
    spellcaster.reparo()
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
```

8. Update `spellcaster.py` as necessary to accomodate your new spells/spell names