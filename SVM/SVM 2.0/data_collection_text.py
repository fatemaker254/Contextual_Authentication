import csv
import time
from pynput import keyboard

OUTPUT_FILE = "raw_keystrokes_new.csv"

# Fixed sentence (NOT stored in CSV)
TARGET_SENTENCE = "biometric authentication is secure"

class SessionRecorder:
    def __init__(self, username):
        self.username = username
        self.session_id = 0
        self.current_session = []
        self.typed_text = ""

    def on_press(self, key):
        try:
            char = key.char
            ascii_val = ord(char)
            self.typed_text += char

        except AttributeError:
            if key == keyboard.Key.space:
                ascii_val = 32
                self.typed_text += " "
            elif key == keyboard.Key.enter:
                self.validate_and_save()
                return
            elif key == keyboard.Key.backspace:
                self.typed_text = self.typed_text[:-1]
                return
            elif key == keyboard.Key.esc:
                print("\nStopping...")
                return False
            else:
                return

        timestamp = int(time.time() * 1000)

        self.current_session.append(
            (self.username,
             self.session_id,
             ascii_val,
             "Down",
             timestamp)
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
            (self.username,
             self.session_id,
             ascii_val,
             "Up",
             timestamp)
        )

    def validate_and_save(self):

        if self.typed_text.strip() == TARGET_SENTENCE:

            if len(self.current_session) > 5:
                with open(OUTPUT_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(self.current_session)

                print(f"Session {self.session_id} saved.")
                self.session_id += 1

        else:
            print("Sentence mismatch! Session discarded.")

        # Reset for next attempt
        self.current_session = []
        self.typed_text = ""

    def start(self):
        print("\nType the following sentence exactly:")
        print(f"\"{TARGET_SENTENCE}\"")
        print("\nPress ENTER to submit. ESC to stop.\n")

        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as listener:
            listener.join()


if __name__ == "__main__":
    user = input("Enter username: ")
    recorder = SessionRecorder(user)
    recorder.start()