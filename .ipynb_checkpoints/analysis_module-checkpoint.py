"""
This module contains common helper functions used for compiling and analysing webcam eye-tracking trial data.
For more details refer to our project and pre-registration at https://osf.io/qh8kx/

Author: Shreshth Saxena (shreshth.saxena@ae.mpg.de)
"""


import pandas as pd
import numpy as np    
import imutils
import math
import cv2
from scipy.stats.mstats import winsorize
from datetime import datetime as dt
from enum import Enum
from tqdm import tqdm

##globals
from enum import Enum
class pred_path(Enum):
    ETH = "csv_backup/example_trials/ETHXGAZE"
    MPII = "csv_backup/example_trials/MPIIGAZE"
    FAZE = "csv_backup/example_trials/FAZE"
    EAR = "csv_backup/example_trials/EAR"
    RT_BENE = "csv_backup/example_trials/RT_BENE"
    
__fnames = ['i05june05_static_street_boston_p1010806',
         'i102423191',
         'i110996888',
         'i1126243635',
         'i1142164052',
         'i1158892521',
         'i117772445',
         'i12030916',
         'i12049788',
         'i132419257',
         'i14020903',
         'i1508828',
         'i2057541',
         'i2234959271',
         'i40576393',
         'i4466881']
FV_IMAGES = {i:f for i,f in zip(range(1,17),__fnames)}

## UTIL FUNCTIONS
def get_dist(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def fu_to_deg(size):
    return size/54.05

def get_frame_count(fname):
    vid = cv2.VideoCapture(fname)
    c=0
    while True:
        ret = vid.grab()
        if not ret:
            break
        c+=1
    return c

def get_time(row):
    format = '%Y-%m-%dT%H:%M:%S.000Z'
    start = dt.strptime(row["start_time"],format)
    end = dt.strptime(row["end_time"],format)
    duration = end-start
    return duration, row["completed"]


def time_to_frame(l, fps):
    return [int(fps*float(pt)) for pt in l]

def frame_to_time(l, fps):
    if type(l) == list:
        return [int(frame/fps) for frame in l]
    else:
        return int(l/fps)

def circ_winsorize(reg_angles, ref):
    rotated = [i-ref for i in reg_angles] #center around ref angle
    rotated = neg_180_to_180(rotated) #convert to range (-180 , 180)
    win_angles = winsorize(rotated, limits=[0.1,0.1])
    ret_arr = [(i+360) if i<0 else i for i in win_angles] #convert to range (0,360)
    ret_arr = [i+ref for i in ret_arr] #rotate back
    ret_arr = [i-360 if i>360 else i for i in ret_arr] #overflow check 
    return ret_arr
    
def neg_180_to_180(a):
    a = [(x-360) if x>180 else x for x in a]
    a = [(x+360) if x<-180 else x for x in a]
    return a

def clip(i,max):
    if i > max:
        return max
    elif i < 0:
        return 0
    else:
        return i

def flat_list(lis):
    return np.array([i for j in list(lis) for i in j])
    
def get_zone(x,y,grid_size):
    x = clip(x,1599)
    y = clip(y,899)
    c = x//(1600//grid_size[0])
    r = y//(900//grid_size[1])
    return int(c+(r*grid_size[1]))

def zone_center(pt):
    center_x = (1600/4) * ((pt-1)%4) #zone width * column
    center_y = (900/4) * ((pt-1)//4) #zone height *row
    return (center_x,center_y)

def labVanced_present(img):
    #LabVanced frame size = (1600,900)
    if (img.shape[1]/img.shape[0]) <= 16/9 : #width/height
        img = imutils.resize(img,height = 900)
        offset = (1600-img.shape[1])//2
        xbounds = (offset,offset+img.shape[1])
        ybounds = (0,900)
    else:
        print("RESIZING ON WIDTH") #debug
        img = imutils.resize(img,width = 1600)
        offset = (900-img.shape[0])//2
        xbounds = (0,1600)
        ybounds = (offset,offset+img.shape[0])
#         temp=(csv-csv.min())/(csv.max()-csv.min())
#         temp["poly_x"] = temp["poly_x"]*1600
#         temp["poly_y"] = temp["poly_y"]*900
    return img,xbounds,ybounds


## from retinaface import RetinaFace (not used finally but a nice face detection alternative)
# def face_n_fps(path, vid_len):
#     vid = cv2.VideoCapture(path)
#     c,f=0,0
#     while True:
#         ret,frame = vid.read()
#         resp = RetinaFace.detect_faces(frame)
#         if not ret:
#             break
#         if len(resp.keys()) > 0:
#             f+=1
#         c+=1
#         print(c)
#     return f*100/c, c/float(vid_len)


## consolidate Calib Tests Dataframe functions for calib_tests.ipynb
def consolidate_1(subjects, model, labels = ["Beg", "Beg+Mid", "Beg+Mid+End"], save = False):
    ct1 = pd.DataFrame(columns = ["subject","factor","acc"])
    for subb in tqdm(subjects): 
        fix_analyse = Fixation(subb, show = False)
        for factor in [1,2,3]:
            trial_x, trial_y, *_ = fix_analyse.parse_trials(model, f"poly_x_{factor}", f"poly_y_{factor}", calib_test=1, show = False)
            acc = get_fix_acc(fix_analyse.gt_points, trial_x, trial_y)
            ct1 = ct1.append({"subject":subb, "factor": labels[factor-1], "acc":acc}, ignore_index = True)
    if save: ct1.to_csv(f"calib_tests_df/{model.name.lower()}_ct1.csv", index = False)
    return ct1

def consolidate_2(subjects, model, labels = ["Beg+Mid+End", "BlockWise"],save = False):
    ct2 = pd.DataFrame(columns = ["subject","factor","acc"])
    for subb in tqdm(subjects): 
        fix_analyse = Fixation(subb, show = False)
        for test,col in zip([1,2], [("poly_x_3", "poly_y_3"), ("poly_x","poly_y")]): #UPDATE from best results of calib_test1
            trial_x, trial_y, *_ = fix_analyse.parse_trials(model, col[0], col[1], calib_test=test, show = False)
            acc = get_fix_acc(fix_analyse.gt_points, trial_x, trial_y)
            ct2 = ct2.append({"subject":subb, "factor": labels[test-1], "acc":acc}, ignore_index = True)
    if save: ct2.to_csv(f"calib_tests_df/{model.name.lower()}_ct2.csv", index = False)
    return ct2

def consolidate_3(subjects, model, labels=["E","SP","E+SP"], save = False):
    ct3 = pd.DataFrame(columns = ["subject","factor","acc"])
    for subb in tqdm(subjects): 
        fix_analyse = Fixation(subb, show = False)
        for factor in [1,2,3]: 
            trial_x, trial_y, *_ = fix_analyse.parse_trials(model, f"poly_x_{factor}", f"poly_y_{factor}", calib_test=3, show = False)
            acc = get_fix_acc(fix_analyse.gt_points, trial_x, trial_y)
            ct3 = ct3.append({"subject":subb, "factor": labels[factor-1], "acc":acc}, ignore_index = True)
    if save: ct3.to_csv(f"calib_tests_df/{model.name.lower()}_ct3.csv", index = False)
    return ct3