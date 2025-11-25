import pyautogui
import time

while True:
    pyautogui.mouseDown(x=1, y=1)
    pyautogui.click()
    pyautogui.mouseUp(x=1, y=1)
    print("clicked")
    time.sleep(10)