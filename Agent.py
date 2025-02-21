from mss import mss # screen captures
import pydirectinput #inputs into computer
import cv2 # frame processing
import numpy as np # math library/multipurpose
import pytesseract # translate pictures into text, important for actions defined in environment
                   # for example, taking a capture of the game over screen, turning it into text -> making an action
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #you may need to set this for your machine
from matplotlib import pyplot as plt
import time # allows us to buffer and pause model
from gym import Env
from gym import spaces
from gym.spaces import Box, Discrete
from YOLO_Object_Detection import *
from pynput.mouse import Controller
import threading
import time

import win32api
import win32con
import pyautogui
import pygetwindow as gw

from pynput.keyboard import Controller, Key


from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback



class GeoManiaEnv(Env):
    def __init__(self):

        self.VK_CODE = {
        'w': 0x57,
        'a': 0x41,
        's': 0x53,
        'd': 0x44,
    }
        self.enemies = None
        self.keyboard = Controller()
        super().__init__()
        self.mouse = Controller()
        self.MAX_ENEMIES = 10
        self.shooting = False
        self.last_position = None
        self.shoot_bool = False
        self.window_name = "GeoMania by NIk_Fot - Google Chrome"
        self.cfg_file_name = r"C:\Users\brice\Downloads\yolo-opencv-detector-main (1)\yolo-opencv-detector-main\yolov4-tiny\yolov4-tiny-custom.cfg"
        self.weights_file_name = r"C:\Users\brice\Downloads\yolo-opencv-detector-main (1)\yolo-opencv-detector-main\yolov4-tiny-custom_last (1).weights"
        self.enemies = None
        self.wincap = WindowCapture(self.window_name)
        self.improc = ImageProcessor(self.wincap.get_window_size(), self.cfg_file_name, self.weights_file_name)


        threading.Thread(target=self.async_enemy_detection, daemon=True).start()
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0,high=255,shape=(640, 480, 3), dtype=np.uint8), # here we are also going to assume only image data, we may need to increase the size of this later on
            "enemies": spaces.Sequence(spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32))  # Each enemy: (x, y)
    })# this will represent the enemies being observed      
        
        self.action_space = spaces.MultiDiscrete([ #this more complex discrete map will allow us to create a more complex set of actions that a model will take. This is subject to change as time goes on
            5, # movement, this will include 5 actions: 0 = no move, 1 = up, 2 = down, 3 = left, 4 = right
            2, # shooting, which will include 3 actions: 0 = no_shoot, 1 = shoot
            #2, # store, which has 2 actions: buy, no_buy
            2, # grenade, which has 2 actions: grenade, no_grenade
            #2, # shield, which has 2 acition: shield, no_shield LATER WE WILL ADD THE REST IN
            ])
        self.capture = mss()
        self.player_capture = {'top':350, 'left':550, 'width':640, 'height':520}
        self.game_over_capture =  {'top':425, 'left':750, 'width':400, 'height':75}
        self.play_again_button =  {'top':725, 'left':525, 'width':400, 'height':75}
    def async_enemy_detection(self):
        while True:
            self.enemies = enemy_detection_positions(self.window_name, self.cfg_file_name, self.weights_file_name, self.wincap, self.improc)
            time.sleep(0.05)
    def bring_window_to_front(self):
        window = gw.getWindowsWithTitle(self.window_name)
        if window:
            win = window[0]
            win.activate()
    def get_observation(self):
        raw_image =  np.array(self.capture.grab(self.player_capture))[:,:,:3] # raw image matrix of capture, with only 3 channels, as the 4th is not needed
        gray_scale = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY) #scaling to gray scale for binary color scheme
        resize = cv2.resize(gray_scale, (640,520))
        channel = np.reshape(resize, (1,520,640)) #stupid resize things for model later on i guess

        return channel
      
    
    def game_over(self):
        game_over_capture = np.array(self.capture.grab(self.game_over_capture))[:,:,:3] #same as before, but grabbing only the "You Died" text
        done = False
        gray_scale = cv2.cvtColor(game_over_capture, cv2.COLOR_BGR2GRAY) #Gray scale to improve Optical Character Recognition software
        cap = pytesseract.image_to_string(gray_scale)[:3] #grabbing only the word "You", as for some reaon it reads "You Died" as "Youd leop"
        if cap == "You":
            done = True
        return cap, gray_scale, done
    
    def play_again(self):
        play_again_capture = np.array(self.capture.grab(self.play_again_button))[:,:,:3]
        return play_again_capture

    def step(self, action):
         
        self.take_action(action)
        text, capture, done = self.game_over() # need to check if the game is over
        new_observation = self.get_observation() # grabbing next capture
        reward = 1
        info = {}
        return new_observation, reward, done, info



    def take_action(self, action):
        movement, shooting, grenade = action  # Extract movement and shooting actions
        active_keys = set()

        # Movement Handling
        if movement == 1:
            active_keys.add('w')
        elif movement == 2:
            active_keys.add('s')
        elif movement == 3:
            active_keys.add('a')
        elif movement == 4:
            active_keys.add('d')

        

        # Press active movement keys
        for key in active_keys:
            self.keyboard.press(key)  # Key down
            time.sleep(0.1)  # Small delay between key presses

        # Release non-active movement keys
        for key in ['w', 'a', 's', 'd']:
            if key not in active_keys:
                self.keyboard.release(key)  # Key up
                time.sleep(0.1)  # Small delay between key releases

        # Shooting Handling
        if shooting == 1 and not self.shoot_bool:
            self.shoot_bool = True
            pyautogui.mouseDown()  # Mouse down (shoot)
        elif shooting == 0 and self.shoot_bool:
            self.shoot_bool = False
            pyautogui.mouseUp()  # Mouse up (stop shooting)

        # Move Mouse to Enemy
        if self.shoot_bool and self.enemies:
            x, y = self.enemies[0], self.enemies[1]
            if (x, y) != self.last_position:  # Only move if position changed
                pyautogui.moveTo(x, y)
                self.last_position = (x, y)
        if grenade == 1:
            self.keyboard.press(Key.space)  # Spacebar press for grenade
            time.sleep(0.1)  # Small delay
            self.keyboard.release(Key.space) 


    
    def render(self):
        #cv2.imshow('Game', np.array(self.capture.grab(self.player_capture))[:,:,:3])
        pass
    def close(self):
        cv2.destroyAllWindows()
    def reset(self):
        time.sleep(1)
        pydirectinput.moveTo(self.play_again_button['top'], self.play_again_button['left'] + 250)
    
    # Simulate the click (left mouse button)
        pydirectinput.click()

    # Optionally, wait for the game to reset or for a short delay
        time.sleep(1)
        return self.get_observation()
       


# Wrap your environment to work with stable-baselines3
