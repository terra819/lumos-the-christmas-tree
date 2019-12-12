import time
import random
import opc
import os
import subprocess
import signal

fadecandyUrl = 'localhost:7890'

# Create a client object
client = opc.Client(fadecandyUrl)
numLEDs = 50

# playable animation sequences
LUMOS = "lumos" 
AGUAMENTI = "aguamenti"
INCENDIO = "incendio"
BROKEN = "broken"

broke = False
current_spell = ''

dir_path = os.path.dirname(os.path.realpath(__file__))
mp3Dir = dir_path + "/mp3s/"
media_prefix= "file://" + mp3Dir
vlc_path = "/usr/bin/cvlc"

def killMusic():
    print("killing music")
    if 'musicPlayer' in vars() or 'musicPlayer' in globals():
        musicPlayer.kill()

def playMusic(file): 
    global musicPlayer
    print(f"playing music {file}")
    killMusic()
    musicPlayer = subprocess.Popen(f"exec {vlc_path} {media_prefix + file}", shell=True)

def killLights():
    print("killing lights")
    pauseLights()
    pixels = [ (0,0,0) ] * numLEDs
    client.put_pixels(pixels)

def playLights(file): 
    global lightPlayer
    print(f"playing lights {file}")
    killLights()
    lightPlayer = subprocess.Popen(f"exec python3 {dir_path}/{file}", shell=True)

def pauseLights():
    print("pausing lights")
    if 'lightPlayer' in vars() or 'lightPlayer' in globals():
        lightPlayer.kill()

def setNewSpell(spell):
    global current_spell, previous_spell, paused
    print("setting new spell: " + spell)
    previous_spell = current_spell
    current_spell = spell
    killMusic()
    killLights()
    paused = False

def lumos():
    print("Lumos called")
    if broke:
        print("broke")
        return
    setNewSpell(LUMOS)

    # bright yellow light
    pixels = [ (255,255,0) ] * numLEDs
    client.put_pixels(pixels)

def nox():
    print("Nox called")
    if broke:
        print("broke")
        return
    #turn off lights
    killLights()

def aguamenti():
    print("Aguamenti called")
    if broke:
        print("broke")
        return
    setNewSpell(AGUAMENTI)

    #run water animation and sounds
    playMusic("hptheme.mp3")
    playLights("lights_aguamenti.py")

def finite_incantatem():
    print("Finite Incantatem called")
    if broke:
        print("broke")
        return
    #turn off lights and music
    killMusic()
    killLights()
    
def arresto_momentum():
    print("Arresto Momentum called")
    global paused
    if broke:
        print("broke")
        return
    #play record scratch
    playMusic("hptheme.mp3")
    # don't need to stop the player, it shold only be a couple of seconds long
    #pause current animation and music
    pauseLights()

def silencio():
    print("Silencio called")
    if broke:
        print("broke")
        return
    #turn off music
    killMusic()

def incendio():
    print("Incendio called")
    if broke:
        print("broke")
        return
    #play fire animation and sounds
    setNewSpell(INCENDIO)
    playMusic("hptheme.mp3")
    playLights("lights_incendio.py")

def reparo():
    print("reparo called")
    if not broke:
        print("not broke")
        return
    #resume current animation
    
    # play starting up sound then restart current spell
    playMusic("hptheme.mp3")
    if previous_spell == LUMOS:
        lumos()
    elif previous_spell == AGUAMENTI:
        aguamenti()
    elif previous_spell == INCENDIO:
        incendio()

def broken():
    print("broken called")
    if not broken:
        print("not broken")
        return
    setNewSpell(BROKEN)
    #play spazzy electric sounds
    playMusic("hptheme.mp3")
    playLights("lights_broken.py")