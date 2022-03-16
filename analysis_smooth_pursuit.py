import os
import pandas as pd
import numpy as np
import ffmpeg
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy.stats.mstats import winsorize
from analysis_module import *
import saccademodel
from scipy.stats import circmean as cim, circstd as cis
from sklearn.linear_model import LinearRegression


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
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/pred_allcalib/Block_{row.Block_Nr}/Smooth Pursuit{row.Trial_Id}.csv"))
            
            for index,pt in enumerate(l):
                sub = pred_df[pred_df.frame.between(pt[0],pt[1])] # movement duration
                trial_x[seq[index]].append(sub[colx])
                trial_y[seq[index]].append(sub[coly])
                try:
                    sub2 = pred_df[pred_df.frame.between(click_frames[index],pt[1])] # from user click to movement stop
                    sm_output = saccademodel.fit([(row.pred_x, row.pred_y) for _,row in sub2.iterrows()])
                    onset_time = frame_to_time([len(sm_output["source_points"])], fps)[0]
                    onsets[seq[index]].append((click_times[index]+onset_time) - start_times[index]) # detected onset time from movement start in ms
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
      
# Analysis functions

def process_trials(trial_x, trial_y, angles, show = False):
    avg = {k:[] for k in angles}
    for angle in trial_x.keys():
        for trial in range(10):
#             sm_x = savgol_filter(trial_x[angle][trial], 7, 1)
#             sm_y = savgol_filter(trial_y[angle][trial], 7, 1)
            if trial_x[angle][trial].shape[0] < 15: #Handling with min trial sample = window_size
                continue
            #rolling mean
            sm_x = trial_x[angle][trial].rolling(15).mean().dropna()
            sm_y = trial_y[angle][trial].rolling(15).mean().dropna()
            
            model = LinearRegression()
            X = np.array(sm_x).reshape(-1,1)
            Y = sm_y
            model.fit(X,Y)
            pred = model.predict(X)                                 
            
            math.degrees(math.atan(model.coef_))
            ang = np.rad2deg(np.arctan2(pred[-1] - pred[0], X[-1] - X[0])) 
            ang = -ang if ang<0 else 360-ang #invert y-axis
            
            if show:                
                plt.scatter(X,Y)
                plt.scatter(trial_x[angle][trial], trial_y[angle][trial], color = "black")
                plt.plot(X,pred, color = "orange")
                plt.gca().invert_yaxis()
                plt.show()
                print("ang ",ang)
            avg[angle].append(ang[0])
    return avg

def mean_angle_preds(trial_x,trial_y, angles, show= False):
    reg_angles = process_trials(trial_x,trial_y, angles)#, show=True)
    cmean,diff,cstd = {},{},{}
    for angle in angles:
        win_angles = circ_winsorize(reg_angles[angle],angle) #winsorize around ref angle
        cmean[angle] = cim(win_angles, high=360, low=0) #Filter Outliers ? trimmed mean oder std rule.... update: implemented circ winsorize
        diff[angle] = abs(cmean[angle]-angle) #only magnitude
        cstd[angle] = cis(win_angles, high=360, low=0)
        if show:
            print("*"*50)
            print("angle: ",angle)
            print("orig: ",np.int64(reg_angles[angle]))
            print("wins: ",np.int64(win_angles))
            print("diff: ", diff[angle] if diff[angle]<180 else 360-diff[angle])
            print("mean: ", cmean[angle])
            print("orig mean: ",cim(reg_angles[angle], high=360, low=0))
            print("stdv: ", cstd[angle])
    return cmean,diff,cstd