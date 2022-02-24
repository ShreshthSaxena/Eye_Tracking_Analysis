import os
import glob
import pandas as pd
import numpy as np
import ffmpeg    
import imutils
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import statistics
from scipy.stats.mstats import winsorize
from enum import Enum
from tqdm import tqdm

##globals
from enum import Enum
class pred_path(Enum):
    ETH = "Models/ETH-XGaze-master/ETH-XGaze-master/Subjects/"
    MPII = "Models/pytorch_mpiigaze-master/Subjects/"
    FAZE = "Models/FAZE/few_shot_gaze/demo/Subjects/"
    EAR = "Models/blink_Soukuova_and_Check/Subjects/"
    RT_BENE = "Models/rt_gene-master/rt_gene-master/rt_bene_standalone/Subjects/"
    
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

def time_to_frame(l, fps):
    return [int(fps*float(pt)) for pt in l]

def frame_to_time(l, fps):
    return [int(frame/fps) for frame in l]

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

def get_zone(x,y,grid_size):
    x = clip(x,1599)
    y = clip(y,899)
    c = x//(1600//grid_size[0])
    r = y//(900//grid_size[1])
    return int(c+(r*grid_size[1]))

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

## Task Measures
def get_fix_acc(GT, trial_x,trial_y):
    error_trials = []
    for trial in range(10):
        error_pts = []
        for pt in range(1,14):
            error_pts.append(get_dist(GT[pt-1], (trial_x[pt][trial],trial_y[pt][trial])))
        assert len(error_pts)==13, print(len(error_pts))
        error_trials.append(winsorize(pd.Series(error_pts).dropna(), limits=[0.1,0.1]).mean())
    acc = winsorize(error_trials, limits=[0.1,0.1]).mean() 
    return acc

def get_fix_precision(rms, std):
    np_rms = rms.copy()
    np_std = std.copy()
    #winsorize mean over points
    for i in range(10):
        np_rms[:,i] = winsorize(np_rms[:,i], limits=[0.1,0.1])
        np_std[:,i] = winsorize(np_std[:,i], limits=[0.1,0.1])
    #mean over trials
    mean_rms = winsorize(np_rms.mean(axis=0), limits=[0.1,0.1]).mean()
    mean_std = winsorize(np_std.mean(axis=0), limits=[0.1,0.1]).mean()
    return mean_rms, mean_std

import numpy as np
import pylab
#Brakel, J.P.G. van (2014). "Robust peak detection algorithm using z-scores". Stack Overflow. Available at: https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362 (version: 2020-11-08).
def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

## Calib Tests Dataframes
def consolidate_1(subjects, model, labels = ["Beg", "Beg+Mid", "Beg+Mid+End"], save = False):
    ct1 = pd.DataFrame(columns = ["subject","factor","acc"])
    for subb in tqdm(subjects): 
        fix_analyse = Fixation(subb, show = False)
        for factor in [1,2,3]:
            trial_x, trial_y, *_ = fix_analyse.parse_trials(model, 1, f"poly_x_{factor}", f"poly_y_{factor}", show = False)
            acc = get_fix_acc(fix_analyse.gt_points, trial_x, trial_y)
            ct1 = ct1.append({"subject":subb, "factor": labels[factor-1], "acc":acc}, ignore_index = True)
    if save: ct1.to_csv(f"calib_tests_df/{model.name.lower()}_ct1.csv", index = False)
    return ct1

def consolidate_2(subjects, model, labels = ["Beg+Mid+End", "BlockWise"],save = False):
    ct2 = pd.DataFrame(columns = ["subject","factor","acc"])
    for subb in tqdm(subjects): 
        fix_analyse = Fixation(subb, show = False)
        for test,col in zip([1,2], [("poly_x_3", "poly_y_3"), ("poly_x","poly_y")]): #UPDATE from best results of calib_test1
            trial_x, trial_y, *_ = fix_analyse.parse_trials(model,test, col[0], col[1], show = False)
            acc = get_fix_acc(fix_analyse.gt_points, trial_x, trial_y)
            ct2 = ct2.append({"subject":subb, "factor": labels[test-1], "acc":acc}, ignore_index = True)
    if save: ct2.to_csv(f"calib_tests_df/{model.name.lower()}_ct2.csv", index = False)
    return ct2

def consolidate_3(subjects, model, labels=["E","SP","E+SP"], save = False):
    ct3 = pd.DataFrame(columns = ["subject","factor","acc"])
    for subb in tqdm(subjects): 
        fix_analyse = Fixation(subb, show = False)
        for factor in [1,2,3]: 
            trial_x, trial_y, *_ = fix_analyse.parse_trials(model,3, f"poly_x_{factor}", f"poly_y_{factor}", show = False)
            acc = get_fix_acc(fix_analyse.gt_points, trial_x, trial_y)
            ct3 = ct3.append({"subject":subb, "factor": labels[factor-1], "acc":acc}, ignore_index = True)
    if save: ct3.to_csv(f"calib_tests_df/{model.name.lower()}_ct3.csv", index = False)
    return ct3

##CLASSES
class Fixation():
    def __init__(self, subb, show=True):
        
        self.gt_points = [(16, 16), (16, 450), (16, 884), (539, 305), (539, 595), (800, 16), (800, 884), (1061, 305), (1061, 595), (1584, 16), (1584, 450), (1584, 884), (800, 450)]
        self.subb = subb
        self.df = pd.read_csv(f"Subjects/{subb}/data.csv")
        self.task_df = self.df[self.df["Task_Name"] == "1. Fixation"]
        if show:
            print("task_df summary \n {}".format(self.summary()))

    def summary(self):
        print(self.task_df.iloc[0].dropna()[8:17])
    
    def parse_trials(self, model, col_x, col_y, pred_final = False, calib_test=None, model_outputs = False, show = True):
        
        trial_x = {key:[] for key in range(1,14)}
        trial_y = {key:[] for key in range(1,14)}
        rms = np.empty((13,10)); rms[:] = np.nan
        std = np.empty((13,10)); std[:] = np.nan
        for _,row in self.task_df.iterrows():
            tr = row.Trial_Id + (5 if row.Block_Nr == 13 else 0)
            fix_seq = [int(i) for i in row.fix_seq.split(";")]
            rec_id = str(row.rec_session_id)
            fname = f"Subjects/{self.subb}/{rec_id}/blockNr_{row.Block_Nr}_taskNr_{row.Task_Nr}_trialNr_{row.Trial_Nr}_FixRec.webm"
            c = get_frame_count(fname)
            vid_len = float(row.RecStop - row.RecStart)
            fps = c/vid_len #in ms
            #vid_len = ffmpeg.probe(fname)["format"]["duration"]  ## using ffprobe, can also use RecStop-RecStart but ffprobe should be better for fps calc .... update: did not work for some subjects so reverting
            
#             if not np.isnan(row.FixRec): #LabVanced wierd behaviour: does not save file names of some recordings
#                 fname = f"Subjects/{self.subb}/{rec_id}/{fixRec}"
            time_pts = row.time_pts.split("---values=")[1].strip("\"").split(";")
            time_pts = [int(t) - int(row.RecStart) for t in time_pts if len(t)>1]
            
            posChange = row.positionChange.split("---values=")[1].strip("\"").split(";")
            posChange = [int(t) - int(row.RecStart) for t in posChange if len(t)>1]
            if len(posChange) == 14:
                posChange = posChange[1:] # first time_pt is when target appears on screen
            
            l = [(int(j),int(i)) for i,j in zip(time_to_frame(posChange,fps),time_to_frame(time_pts, fps))]
            
            if show:
                try:
                    print("Recording length (ffmpeg): ",ffmpeg.probe(fname)["format"]["duration"])
                except:
                    pass
                print("RecStop - RecStart : ",vid_len)
                print("Total Frame Count : ", c)
                print("used Frames Count : ",sum([pt[1]-pt[0] for pt in l]))
                print("FPS : ",fps*1000)
                print("key press times : ",time_pts)
                print("position update times : ",posChange)
                print("diff : ", [int(i)-int(j) for i,j in zip(posChange,time_pts)])
                
            if model_outputs:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/model_outputs/Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
            elif calib_test != None:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/calib_test{calib_test}/outputs/Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
            elif pred_final:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/pred_final/Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
            else:
                continue
            
            for index,pt in enumerate(l):
                sub = pred_df[pred_df.frame.between(pt[0],pt[1])]
                
                # Ehinger et al uses fixation detection algo. to identify last fixation, we're using medians of all points
#                 X = winsorize(sub.poly_x, limits=[0.1,0.1])
#                 Y = winsorize(sub.poly_y, limits=[0.1,0.1])
                try:
                    #accuracy
                    trial_x[fix_seq[index]].append(round(statistics.median(sub[col_x]),2))
                    trial_y[fix_seq[index]].append(round(statistics.median(sub[col_y]),2))
                except:
                    trial_x[fix_seq[index]].append(np.nan)
                    trial_y[fix_seq[index]].append(np.nan)
                    print(f"{self.subb} adding nan to pt {fix_seq[index]} trial {index}") 
                
                #precision
                #RMS
                x_prev = sub[col_x].shift(1)
                y_prev = sub[col_y].shift(1)
                d_sq = (sub[col_x] - x_prev)**2 + (sub[col_y] - y_prev)**2
                rms[fix_seq[index]-1][tr-1] = (math.sqrt(d_sq.mean()))

                #STD
                x_mean = sub[col_x].mean()
                y_mean = sub[col_y].mean()
                d_sq = (sub[col_x]-x_mean)**2 + (sub[col_y]-y_mean)**2
                std[fix_seq[index]-1][tr-1] = (math.sqrt(d_sq.mean()))
                    
                if show: 
                    plt.scatter(sub[col_x],sub[col_y], color = self.palette[fix_seq[index]])
            if show:
                plt.gca().invert_yaxis()
                plt.show()
                print("-"*50)
        return trial_x,trial_y,rms,std

    
import saccademodel
class Smooth_Pursuit():
    def __init__(self, subb, show=True):
        
        self.subb = subb
        self.df = pd.read_csv(f"Subjects/{subb}/data.csv")
        self.task_df = self.df[self.df["Task_Name"] == "2. Smooth Pursuit"]
        self.angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        self.palette = sns.color_palette('colorblind', len(self.angles))
        if show:
            print("sp_df summary \n {}".format(self.summary()))

    def summary(self):
        print(self.task_df.iloc[0].dropna()[7:22])
    
    def parse_trials(self, model,colx = 'pred_x', coly = 'pred_y', show = True, model_outputs = False):
        
        trial_x = {key:[] for key in self.angles}
        trial_y = {key:[] for key in self.angles}
        onsets = {key:[] for key in self.angles}
        for _,row in self.task_df.iterrows():
            
            seq = [int(i) for i in row.angles.split(";")]
            
            rec_id = row.rec_session_id
#             fname = "Subjects/s{}/{}/{}".format(self.subb,rec_id,row.pursuit_rec)
            fname = f"Subjects/{self.subb}/{rec_id}/blockNr_{row.Block_Nr}_taskNr_{row.Task_Nr}_trialNr_{row.Trial_Nr}_pursuit_rec.webm"
            c = get_frame_count(fname)
            vid_len = float(row.RecStop - row.RecStart)
            fps = c/vid_len #in ms
            
            start_times = row.anim_time.split("---values=")[1].strip("\"").split(";")
            start_times = [int(t) - int(row.RecStart) for t in start_times if len(t)>1]
            
            stop_times = row.ResetTimes.split("---values=")[1].strip("\"").split(";")
            stop_times = [int(t) - int(row.RecStart) for t in stop_times if len(t)>1]
            
            click_times = row.ClickTimes.split("---values=")[1].strip("\"").split(";")
            click_times = [int(t) - int(row.RecStart) for t in click_times if len(t)>1]
            
            
            l = [(int(j),int(i)) for i,j in zip(time_to_frame(stop_times, fps),time_to_frame(start_times, fps))]
            click_frames = time_to_frame(click_times, fps)
            
            if show:
                try:
                    print("Recording length (ffmpeg): ",ffmpeg.probe(fname)["format"]["duration"])
                except:
                    pass
                print("RecStop - RecStart : ",vid_len)
                print("Total Frame Count : ",c)
                print("used Frames Count : ",sum([pt[1]-pt[0] for pt in l]))
                print("FPS : ",fps*1000)
                print("start times : ",start_times)
                print("stop times : ",stop_times)
                print("click_times : ", click_times)
                print("diff : ", [int(i)-int(j) for i,j in zip(stop_times,start_times)])
                
            if model_outputs:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/model_outputs/Block_{row.Block_Nr}/Smooth Pursuit{row.Trial_Id}.csv"))
            else:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/pred_final/Block_{row.Block_Nr}/Smooth Pursuit{row.Trial_Id}.csv"))
            
            for index,pt in enumerate(l):
                sub = pred_df[pred_df.frame.between(pt[0],pt[1])] # movement duration
                trial_x[seq[index]].append(sub[colx])
                trial_y[seq[index]].append(sub[coly])
                try:
                    sub2 = pred_df[pred_df.frame.between(click_frames[index],pt[1])] # from user click to movement stop
                    sm_output = saccademodel.fit([(row.pred_x, row.pred_y) for _,row in sub2.iterrows()])
                    onset_time = frame_to_time([len(sm_output["source_points"])], fps)[0]
                    onsets[seq[index]].append((click_times[index]+onset_time) - start_times[index]) # detected onset time from movement start
#                     print(click_times[index]+onset_time, start_times[index])
                except Exception as e:
                    print(e)
                    onsets[seq[index]].append((np.nan, start_times[index]))
                    print(f"{self.subb} adding nan to pt {seq[index]} trial {index}") 
                
                
                
                if show: 
                    plt.scatter(sub[colx],sub[coly], color = self.palette[self.angles.index(seq[index])])
            if show:
#                     plt.xlim(0,max(1600,sub.poly_x.max()))
#                     plt.ylim(0,max(900,sub.poly_y.max()))
                    plt.gca().invert_yaxis()
                    plt.show()
                    print("-"*50)
        return trial_x,trial_y, onsets
    
class Zone_Classification():
    def __init__(self, subb, show=True):
        
        self.subb = subb
        self.task = "4. Zone Classification"
        self.df = pd.read_csv(os.path.join("Subjects",subb,"data.csv"))
        self.task_df = self.df[self.df["Task_Name"] == self.task]
        self.palette = sns.color_palette('colorblind', 16)
        if show:
            print("sp_df summary \n {}".format(self.summary()))

    def summary(self):
        print(self.task_df.iloc[0].dropna()[7:17])
    
    def parse_trials(self, model, show = True, model_outputs = False):
        
        trial_x = {key:[] for key in range(1,17)}
        trial_y = {key:[] for key in range(1,17)}
        
        for _,row in self.task_df.iterrows():
            
            seq = [int(i) for i in row.seq.split(";")]
            
            rec_name = row.recordZone
            rec_id = row.rec_session_id
#             fname = "Subjects/{}/{}/{}".format(self.subb,rec_id,rec_name)
            fname = f"Subjects/{self.subb}/{rec_id}/blockNr_{row.Block_Nr}_taskNr_{row.Task_Nr}_trialNr_{row.Trial_Nr}_recordZone.webm"
            c = get_frame_count(fname)
            vid_len = float(row.RecStop - row.RecStart)
            fps = c/vid_len #in ms
        
        
            start_times = row.pos_update_time.split("---values=")[1].strip("\"").split(";")
            start_times = [int(t) - int(row.RecStart) for t in start_times if len(t)>1]
            
            stop_times = row.pos_hide_time.split("---values=")[1].strip("\"").split(";")
            stop_times = [int(t) - int(row.RecStart) for t in stop_times if len(t)>1][1:] #For Zone Classification discard the first hide time

            if show:
                print("Recording length : ",ffmpeg.probe(fname)["format"]["duration"])
                print("RecStop - RecStart : ",vid_len)
                print("Total Frame Count : ",c)
                print("FPS : ",c/float(vid_len))
                print("start times : ",start_times)
                print("stop times : ",stop_times)
                print("diff : ", [int(i)-int(j) for i,j in zip(stop_times,start_times)])
                
            if model_outputs:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/model_outputs/Block_{row.Block_Nr}/Zone Classification{row.Trial_Id}.csv"))
            else:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/pred_final/Block_{row.Block_Nr}/Zone Classification{row.Trial_Id}.csv"))
                
            l = [(int(j),int(i)) for i,j in zip(time_to_frame(stop_times, fps),time_to_frame(start_times, fps))]
            for index,pt in enumerate(l):
#                 sub = pred_df.iloc[pt[0]:pt[1]]
                sub = pred_df[pred_df.frame.between(pt[0],pt[1])]
#                 X = winsorize(sub.poly_x, limits=[0.1,0.1])
#                 Y = winsorize(sub.poly_y, limits=[0.1,0.1])

                try:
                    trial_x[seq[index]].append(round(statistics.median(sub.pred_x),2))
                    trial_y[seq[index]].append(round(statistics.median(sub.pred_y),2))
                except:
                    trial_x[seq[index]].append(np.nan)
                    trial_y[seq[index]].append(np.nan)
                    print(f"{self.subb} adding nan to zone {seq[index]} trial {index}") 
                
                if show: 
                    plt.scatter(sub.pred_x,sub.pred_y, color = self.palette[seq[index]-1])
            if show:
#                     plt.xlim(0,max(1600,sub.poly_x.max()))
#                     plt.ylim(0,max(900,sub.poly_y.max()))
                    plt.gca().invert_yaxis()
                    plt.show()
                    print("-"*50)
            
            mean_x = {k:winsorize(v, limits=[0.1,0.1]).mean() for k,v in trial_x.items()}
            mean_y = {k:winsorize(v, limits=[0.1,0.1]).mean() for k,v in trial_y.items()}
        return trial_x,trial_y,mean_x,mean_y
    
    
import sys
sys.path.append('/hpc/users/shreshth.saxena/Models/blink_Soukuova_and_Check/')
from quick_blinks import get_ear

class FreeView():
    def __init__(self, subb, show=True):
        
        self.subb = subb
        self.task = "3. FreeView"
        df = pd.read_csv("Subjects/"+subb+"/data.csv")
        self.task_df = df[(df["Task_Name"] == self.task) & (df["Trial_Id"] != 17)]
        self.palette = sns.color_palette("colorblind", 12)
    
    def parse_trials(self, model, show = True, model_outputs = False, analyse_blinks = False):
        
        trial_x = {key:[] for key in range(1,17)}
        trial_y = {key:[] for key in range(1,17)}
        for _,row in self.task_df.iterrows():
            
#             rec_name = row.freeViewRec #could be used but Labvanced data has random Nan entries even when file is present
            rec_id = row.rec_session_id
            fname = f"Subjects/{self.subb}/{rec_id}/blockNr_{row.Block_Nr}_taskNr_{row.Task_Nr}_trialNr_{row.Trial_Nr}_freeViewRec.webm"
            if analyse_blinks:
                temp = get_ear(fname)
                c = temp.shape[0]
                temp.plot(x="frame",y="ear", kind ="line")
            else:
                c = get_frame_count(fname)
            vid_len = float(row.RecStop - row.RecStart)
            fps = c/vid_len #in ms
            
            if show:
                print("Recording length : ",ffmpeg.probe(fname)["format"]["duration"])
                print("RecStop - RecStart : ",vid_len)
                print("Frame Count : ",c)
                print("FPS : ",fps)
                print("file name",fname)
                
            if model_outputs:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/model_outputs/Block_{row.Block_Nr}/FreeView{row.Trial_Id}.csv"))
            else:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/pred_final/Block_{row.Block_Nr}/FreeView{row.Trial_Id}.csv"))
                                    
            if show:
                ax = self.plot_one_subject(row["Trial_Id"],pred_df["pred_x"],pred_df["pred_y"])
                plt.gca().invert_yaxis()
                ax.plot()
                plt.show()
            trial_x[row["Trial_Id"]] = pred_df["pred_x"]
            trial_y[row["Trial_Id"]] = pred_df["pred_y"]
        return trial_x, trial_y
    
    
import ffmpeg   
from scipy.signal import find_peaks, peak_prominences
pd.options.plotting.backend = "matplotlib"

class Blink_Detect():
    def __init__(self, subb, show=True):
        
        self.subb = subb
        self.df = pd.read_csv(f"Subjects/{subb}/data.csv")
        self.task_df = self.df[self.df["Task_Name"] == "5. Blink Detect"]
        self.palette = cm.rainbow(np.linspace(0, 1, 12))
        if show:
            print("blinks_df summary \n {}".format(self.summary()))

    def summary(self):
        print(self.task_df.iloc[0].dropna()[5:17])
    
    def get_preds(self, model):
        trial_probs = {key:[] for key in range(1,11)}
        for _,row in self.task_df.iterrows():
            rec_name = row.BlinkRec
            pred_df = pd.read_csv(os.path.join(model.value, "{}/{}.csv".format(self.subb,rec_name[:-5]))) 
            trial_probs[row.Trial_Id + (5 if row.Block_Nr == 13 else 0)] = pred_df.iloc[:,-1]
        return trial_probs
    
#     def parse_trials(self, model, show = True):
        
#         trial_peaks = {key:[] for key in range(1,11)}
        
#         for _,row in self.task_df.iterrows():
# #             rec_name = row.BlinkRec
#             rec_id = row.rec_session_id
#             rec_name = f"blockNr_{row.Block_Nr}_taskNr_{row.Task_Nr}_trialNr_{row.Trial_Nr}_BlinkRec.webm"
#             fname = f"Subjects/{self.subb}/{rec_id}/{rec_name}"

#             start_times = row.beep_times.split("---values=")[1].strip("\"").split(";")
#             start_times = [int(t) - int(row.RecStart) for t in start_times if len(t)>1]
            
#             stop_times = row.beepEndTimes.split("---values=")[1].strip("\"").split(";")
#             stop_times = [int(t) - int(row.RecStart) for t in stop_times if len(t)>1]
            
#             pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/{rec_name[:-5]}.csv")) 
            
#             # already checked in data screening (see addendum)
# #             if pred_df[pred_df.iloc[:,-1] == "NA"].shape[0] > 0.3*pred_df.shape[0]:
# #                 print("> 30% missing data")
# #                 continue
            
#             disp_df = pred_df.replace("NA",np.nan).interpolate(method='linear') #try other Interpolation methods?
#             med = statistics.median(disp_df.iloc[:,-1])
#             print("median:",med)
#             c = get_frame_count(fname)
#             vid_len = float(row.RecStop - row.RecStart)
#             fps = c/vid_len #in ms
            
#             #threshold conf = 50% and distance between 2 consecutive blinks = 3 frames ~ 3*33 = 99ms since typical blink duration is btw 0.1 to 0.4 s (Ehinger battery paper)
#             if model.name == "EAR":
#                 peaks, _ = find_peaks(-disp_df.iloc[:,-1].to_numpy(), threshold = -1.5*med, distance = 35)
#             elif model.name == "RT_BENE":
#                 peaks, _ = find_peaks(disp_df.iloc[:,-1].to_numpy(), height = 5*med, distance = 35) 
            
#             if show:
#                 print("Recording Name : ",rec_name)
#                 print("Recording length : ",ffmpeg.probe(fname)["format"]["duration"])
#                 print("RecStop - RecStart : ",vid_len)
#                 print("Frame Count : ",c)
#                 print("FPS : ",fps)
#                 print("start times : ",start_times)
#                 print("stop times : ",stop_times)
#                 print("diff : ", [int(i)-int(j) for i,j in zip(start_times, [0]+stop_times[:-1])]) 
#                 disp_df.plot(x="frame", y=disp_df.columns[-1], figsize=(12,8))
#                 ymin,ymax = plt.gca().get_ylim()
#                 plt.vlines(time_to_frame(start_times, fps),ymin,ymax, linestyles ='dashed', colors = 'gray')
#                 plt.plot(peaks, disp_df.iloc[:,-1][peaks], "x", markersize = 10)
#                 plt.show()
#                 print("-"*50)
            
#             trial_peaks[row.Trial_Id + (5 if row.Block_Nr == 13 else 0)] = peaks
            
#         return trial_peaks
    
    def parse_trials_stack(self, model, show = True):
        trial_latency = {key:[] for key in range(1,11)}
        
        for _,row in self.task_df.iterrows():
            
            rec_id = row.rec_session_id
            rec_name = f"blockNr_{row.Block_Nr}_taskNr_{row.Task_Nr}_trialNr_{row.Trial_Nr}_BlinkRec.webm"
            fname = f"Subjects/{self.subb}/{rec_id}/{rec_name}"

            start_times = row.beep_times.split("---values=")[1].strip("\"").split(";")
            start_times = [int(t) - int(row.RecStart) for t in start_times if len(t)>1]
            
            intervals = [int(i)-int(j) for i,j in zip(start_times[1:], start_times[:-1])]
            
            pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/{rec_name[:-5]}.csv"))
            
#             stop_times = row.beepEndTimes.split("---values=")[1].strip("\"").split(";")
#             stop_times = [int(t) - int(row.RecStart) for t in stop_times if len(t)>1]
                        
            if pred_df[pred_df.iloc[:,-1] == "NA"].shape[0] > 0.3*pred_df.shape[0]:
                print("> 30% missing data")
                continue
            
            disp_df = pred_df.replace("NA",np.nan).interpolate(method='linear') #try other Interpolation methods?
#             med = statistics.median(disp_df.iloc[:,-1])
#             print("median:",med)
            
            #threshold conf = 50% and distance between 2 consecutive blinks = 3 frames ~ 3*33 = 99ms since typical blink duration is btw 0.1 to 0.4 s (Ehinger battery paper)
#             if model.name == "EAR":
#                 peaks, _ = find_peaks(-disp_df.iloc[:,-1].to_numpy(), threshold = -1.5*med, distance = 35)
#             elif model.name == "RT_BENE":
#                 peaks, _ = find_peaks(disp_df.iloc[:,-1].to_numpy(), height = 5*med, distance = 35) 
            
            c = get_frame_count(fname)
            vid_len = float(row.RecStop - row.RecStart)
            fps = c/vid_len #in ms
            out = thresholding_algo(disp_df.iloc[:,-1].to_numpy(), 10, 4, 0.3)
            signals = np.pad(out["signals"][40:],(40,0),constant_values=0)
            onsets = time_to_frame(start_times, fps)
            assert len(signals) == disp_df.shape[0]
            
            latency=[]
            mean_interval = time_to_frame([statistics.mean(intervals)],fps)[0]
#             print("mean int",mean_interval) = o/p is in range (45,48)
            for i,j in zip(onsets,(onsets[1:]+[onsets[-1]+mean_interval])):
                if -1 in signals[i:j]:
                    latency.append(np.where(signals[i:j]==-1)[0][0])
                else:
                    latency.append(np.nan)
#             print("LATENCY",latency)
                           
            if show:
                print("Recording Name : ",rec_name)
                print("Recording length : ",ffmpeg.probe(fname)["format"]["duration"])
                print("RecStop - RecStart : ",vid_len)
                print("Frame Count : ",c)
                print("FPS : ",fps)
                print("start times : ",start_times)
#                 print("stop times : ",stop_times)
                print("inter-beat duration : ", intervals) 
                print("Latency: ",latency)
                disp_df.plot(x="frame", y=disp_df.columns[-1], figsize=(12,8))
                ymin,ymax = plt.gca().get_ylim()
                plt.vlines(time_to_frame(start_times, fps),ymin,ymax, linestyles ='dashed', colors = 'gray')
                plt.show()
                plt.figure(figsize=(12,7))
                plt.plot(signals)
                plt.show()
                print("-"*50)
            
            trial_latency[row.Trial_Id + (5 if row.Block_Nr == 13 else 0)] = np.array(latency)
            
        return trial_latency