import numpy as np
import win32gui, win32ui, win32con
from PIL import Image
from time import sleep
import cv2 as cv
import os
import random
from time import time

#Credit goes to https://github.com/moises-dias/yolo-opencv-detector, which is the set of notebooks that aided in creating a dataset, training, and running the model for object detection


# Run this cell to initiate detections using the trained model.


class WindowCapture: #this is the way images are taken, this updates every frame
    w = 0
    h = 0
    hwnd = None

    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

    def get_screenshot(self):
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[...,:3]
        img = np.ascontiguousarray(img) 
            
        return img

    def generate_image_dataset(self):
        if not os.path.exists("images"):
            os.mkdir("images")
        while(True):
            img = self.get_screenshot()
            im = Image.fromarray(img[..., [2, 1, 0]])
            im.save(f"./images/img_{len(os.listdir('images'))}.jpeg")
            sleep(1)
    
    def get_window_size(self):
        return (self.w, self.h)
    
class ImageProcessor: #this processes the set of images and establishes the bounds for what enemies are 
    W = 0
    H = 0
    net = None
    ln = None
    classes = {}
    colors = []

    def __init__(self, img_size, cfg_file, weights_file):
        np.random.seed(42)
        self.net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.W = img_size[0]
        self.H = img_size[1]
        
        with open(r'C:\Users\brice\Downloads\yolo-opencv-detector-main\yolo-opencv-detector-main\yolov4-tiny\obj.names', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            self.classes[i] = line.strip()
        
        # If you plan to utilize more than six classes, please include additional colors in this list.
        self.colors = [
            (0, 0, 255), 
            (0, 255, 0), 
            (255, 0, 0), 
            (255, 255, 0), 
            (255, 0, 255), 
            (0, 255, 255)
        ]
        

    def proccess_image(self, img):

        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)
        outputs = np.vstack(outputs)
        
        coordinates = self.get_coordinates(outputs, 0.5)

        self.draw_identified_objects(img, coordinates)

        return coordinates

    def get_coordinates(self, outputs, conf):

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([self.W, self.H, self.W, self.H])
                p0 = int(x - w//2), int(y - h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)

        if len(indices) == 0:
            return []

        coordinates = []
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            coordinates.append({'x': x, 'y': y, 'w': w, 'h': h, 'class': classIDs[i], 'class_name': self.classes[classIDs[i]]})
        return coordinates

    def draw_identified_objects(self, img, coordinates):
        for coordinate in coordinates:
            x = coordinate['x']
            y = coordinate['y']
            w = coordinate['w']
            h = coordinate['h']
            classID = coordinate['class']
            
            color = self.colors[classID]
            
            #cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            #cv.putText(img, self.classes[classID], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #cv.imshow('window',  img)




window_name = "GeoMania by NIk_Fot - Google Chrome"
cfg_file_name = r"C:\Users\brice\Downloads\yolo-opencv-detector-main (1)\yolo-opencv-detector-main\yolov4-tiny\yolov4-tiny-custom.cfg"
weights_file_name = r"C:\Users\brice\Downloads\yolov4-tiny-custom_last (2).weights"

wincap = WindowCapture(window_name)
improc = ImageProcessor(wincap.get_window_size(), cfg_file_name, weights_file_name)

def enemy_detection_positions(window_name, cfg_file, weights_file, wincap,improc, decay_time=3):
    #We will want ot wrap this in its own method so it can be run at the same time the model begins running also. 
   while(True):
    enemies = []
    player = []
    current_time = time()
    last_known_positions = {}
    ss = wincap.get_screenshot()
    
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

    coordinates = improc.proccess_image(ss)
    
    for coordinate in coordinates:
        if coordinate["class_name"] == "Enemy":
                center_x = int(coordinate["x"] + coordinate["w"] // 2) #this is to find teh center of the bounded box
                center_y  = int(coordinate["y"] + coordinate["h"] // 2)

                last_known_positions[(center_x, center_y)] = current_time
                enemies.append((center_x,center_y))
        if coordinate["class_name"] == "Player":
            center_x_player = int(coordinate["x"] + coordinate["w"] // 2)
            center_y_player = int(coordinate["y"] + coordinate["h"] // 2)
            player.append((center_x_player, center_y_player))
    expired_keys = [pos for pos, timestamp in last_known_positions.items() if current_time - timestamp > decay_time]
    for key in expired_keys:
        del last_known_positions[key]
    all_enemies = list(set(enemies) | set(last_known_positions.keys()))
        # Merge current detections with last known ones
    closest_enemy = find_closest_enemy(all_enemies, player)
    return closest_enemy
        # If you have limited computer resources, consider adding a sleep delay between detections.
        # sleep(0.2)



def find_closest_enemy(enemies,player): #here we can use Euclidean distance to solve
    if not enemies or not player:
        return None
    distances = np.linalg.norm(enemies - np.array([player[0][0], player[0][1]]), axis=1)


    closest_index = np.argmin(distances)
    

    return enemies[closest_index]
