import os
import subprocess
import time
import signal

dir_path = os.path.dirname(os.path.realpath(__file__))
mp3Dir = dir_path + "/mp3s/"
hpThemesong = mp3Dir + "hptheme.mp3"

vlc_path = "/usr/bin/cvlc"
net_stream = "file://" + hpThemesong


playCmd = f"{vlc_path} {net_stream}"
print(playCmd)
p = subprocess.Popen(playCmd, shell=True)
print(p)
time.sleep(10)

print("Quitting...")
os.killpg(os.getpgid(p.pid), signal.SIGTERM)