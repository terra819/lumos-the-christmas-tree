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
paused = False

dir_path = os.path.dirname(os.path.realpath(__file__))
mp3Dir = dir_path + "/mp3s/"
media_prefix= "file://" + mp3Dir
vlc_path = "/usr/bin/cvlc"

def killMusic():
    if 'player' in vars() or 'player' in globals():
        if os.getpgid(player.pid)> 0:
            os.killpg(os.getpgid(player.pid), signal.SIGTERM)

def playMusic(file): 
    global player
    killMusic()
    player = subprocess.Popen(f"{vlc_path} {media_prefix + file}", shell=True)

def killLights():
    global paused
    pixels = [ (0,0,0) ] * numLEDs
    client.put_pixels(pixels)
    paused = True

def setNewSpell(spell):
    global current_spell, previous_spell, paused
    print("setting new spell: " + spell)
    previous_spell = current_spell
    current_spell = spell
    killMusic()
    killLights()
    paused = False

def lumos():
    global player
    print("Lumos called")
    if broke:
        print("broke")
        return
    setNewSpell(LUMOS)

    #turn on red green chasers, hp theme song
    playMusic("hptheme.mp3")
    while current_spell == LUMOS and not paused:
        my_pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] * numLEDs
        random.shuffle(my_pixels)
        client.put_pixels(my_pixels)
        time.sleep(0.3)

def nox():
    print("Nox called")
    if broke:
        print("broke")
        return
    #turn off lights
    killLights()

def aguamenti():
    print("Aguamenti called")
    global player
    if broke:
        print("broke")
        return
    setNewSpell(AGUAMENTI)
    print(paused)

    #run water animation and sounds
    playMusic("water.mp3")
    while current_spell == AGUAMENTI and not paused:
        my_pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] * numLEDs
        random.shuffle(my_pixels)
        client.put_pixels(my_pixels)
        time.sleep(0.3)

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
    #play record scratch then
    # playMusic("recordscratch.mp3")
    # don't need to stop the player, it shold only be a couple of seconds long
    #pause current animation and music
    paused = True

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
    # playMusic("fire.mp3")
    while current_spell == INCENDIO and not paused:
        my_pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] * numLEDs
        random.shuffle(my_pixels)
        client.put_pixels(my_pixels)
        time.sleep(0.3)

def reparo():
    print("reparo called")
    if not broke:
        print("not broke")
        return
    #resume current animation
    
    # play starting up sound then restart current spell
    # playMusic("comingbackonline.mp3")
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
    # playMusic("spazzymusic.mp3")
    # play flicker-y animation that spazzes out
    while current_spell == BROKEN and not paused:
        my_pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] * numLEDs
        random.shuffle(my_pixels)
        client.put_pixels(my_pixels)
        time.sleep(0.3)