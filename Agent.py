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

class GeoManiaEnv(Env):
    def __init__(self):
        self.enemies = None
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
            #2, # grenade, which has 2 actions: grenade, no_grenade
            #2, # shield, which has 2 acition: shield, no_shield LATER WE WILL ADD THE REST IN
            ])
        self.capture = mss()
        self.player_capture = {'top':350, 'left':550, 'width':640, 'height':520}
        self.game_over_capture =  {'top':425, 'left':750, 'width':400, 'height':75}
        self.play_again_button =  {'top':725, 'left':525, 'width':400, 'height':75}
    def async_enemy_detection(self):
        while True:
            self.enemies = enemy_detection_positions(self.window_name, self.cfg_file_name, self.weights_file_name, self.wincap, self.improc)
    def get_observation(self):
        raw_image =  np.array(self.capture.grab(self.player_capture))[:,:,:3] # raw image matrix of capture, with only 3 channels, as the 4th is not needed
        gray_scale = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY) #scaling to gray scale for binary color scheme
        resize = cv2.resize(gray_scale, (640,520))
        channel = np.reshape(resize, (1,520,640)) #stupid resize things for model later on i guess

        print(self.enemies)
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
        movement, shooting = action #for now we will keep only move and shoot and add the rest later
        if movement == 1:
            pydirectinput.keyDown('w')
        elif movement == 2:
            pydirectinput.keyDown('s')
        elif movement == 3:
            pydirectinput.keyDown('a')
        elif movement == 4:
            pydirectinput.keyDown('d')

        if shooting == 1:
            self.shoot_bool = True
        else:
            self.shoot_bool = False
            pydirectinput.mouseDown()

        if self.shoot_bool and self.enemies:  # Ensure it's not empty
            print("movingmouse")
            self.mouse.position = (self.enemies[0], self.enemies[1])
            pydirectinput.mouseDown()


        pydirectinput.keyUp('w') #to make sure the model doesnt keep holding it down, we may change this later
        pydirectinput.keyUp('s')
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('d')
    
    def render(self):
        cv2.imshow('Game', np.array(self.capture.grab(self.player_capture))[:,:,:3])

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
       

env = GeoManiaEnv()

for epoch in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        print(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward