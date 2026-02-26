import csv
import time
from pynput import keyboard

# File path
userFilePath = "Collecting_keyStorke.csv"

# Global username
userName = ""
flag=0
with open(userFilePath, "r") as f:
    user = csv.reader(f)
    header=next(user)
    if header[0] == "user":
        flag=1
    else:
        with open(userFilePath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["user", "key", "keyEvent", "Time"])

class KeyLogger:
    def __init__(self):
        self.eventList = []
        self.isCaps = False
        self.listener = None

    def on_press(self, key):
        try:
            char = key.char
            ascii_val = ord(char)

        # Print character immediately
            print(char, end='', flush=True)

            if self.isCaps and 97 <= ascii_val <= 122:
                ascii_val = ord(char.upper())

        except AttributeError:
            if key == keyboard.Key.space:
                print(" ", end='', flush=True)
                ascii_val = 32
    
            elif key == keyboard.Key.enter:
                print()
                ascii_val = 13

            elif key == keyboard.Key.caps_lock:
                self.isCaps = not self.isCaps
                return

            elif key == keyboard.Key.esc:
                print("\nStopping key capture...")
                self.stop_listener()
                return False

            else:
                ascii_val = 0

        timestamp = int(time.time() * 1000)
        self.storeEvent("Down", ascii_val, timestamp)


    def on_release(self, key):
        try:
            ascii_val = ord(key.char)
        except AttributeError:
            ascii_val = 0

        timestamp = int(time.time() * 1000)
        self.storeEvent("Up", ascii_val, timestamp)

    def storeEvent(self, activity, ascii_val, timestamp):
        self.eventList.append((userName, ascii_val, activity, timestamp))

    def stop_listener(self):
        userRecordData(self.eventList)
        if self.listener:
            self.listener.stop()

    def start(self):
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as self.listener:
            self.listener.join()


def userRecordData(eventList):
    print("\nOutput:")
    print(eventList)

    with open(userFilePath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(eventList)


def getUserName():
    global userName
    userName = input("Enter your Name: ")


def getKeyStroke():
    keyLogger = KeyLogger()
    print("Enter your text (Press ESC to stop):")
    keyLogger.start()


# ---- MAIN ----
if __name__ == "__main__":
    getUserName()
    getKeyStroke()
