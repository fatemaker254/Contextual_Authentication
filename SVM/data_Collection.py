import csv
import time
from pynput import keyboard

OUTPUT_FILE = "raw_keystrokes.csv"

class SessionRecorder:
    def __init__(self, username):
        self.username = username
        self.session_id = 0
        self.current_session = []

    def on_press(self, key):
        try:
            ascii_val = ord(key.char)
        except AttributeError:
            if key == keyboard.Key.space:
                ascii_val = 32
            elif key == keyboard.Key.enter:
                self.save_session()
                return
            elif key == keyboard.Key.esc:
                print("\nStopping...")
                return False
            else:
                return

        timestamp = int(time.time() * 1000)
        self.current_session.append(
            (self.username, self.session_id,
             ascii_val, "Down", timestamp)
        )

    def on_release(self, key):
        try:
            ascii_val = ord(key.char)
        except AttributeError:
            if key == keyboard.Key.space:
                ascii_val = 32
            else:
                return

        timestamp = int(time.time() * 1000)
        self.current_session.append(
            (self.username, self.session_id,
             ascii_val, "Up", timestamp)
        )

    def save_session(self):
        if len(self.current_session) > 5:
            with open(OUTPUT_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.current_session)
            print(f"Session {self.session_id} saved.")
            self.session_id += 1
        self.current_session = []

    def start(self):
        print("Type password and press ENTER. ESC to stop.")
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as listener:
            listener.join()


if __name__ == "__main__":
    user = input("Enter username: ")
    recorder = SessionRecorder(user)
    recorder.start()