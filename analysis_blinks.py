import os
import pandas as pd
import numpy as np
import pylab
import ffmpeg
import matplotlib.pyplot as plt
import statistics
from scipy.stats.mstats import winsorize
from scipy.signal import find_peaks, peak_prominences
from analysis_module import *


class Blink_Detect():
    def __init__(self, subb, show=True):
        
        self.subb = subb
        self.df = pd.read_csv(f"Subjects/{subb}/data.csv")
        self.task_df = self.df[self.df["Task_Name"] == "5. Blink Detect"]
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
        assert model in [pred_path.RT_BENE, pred_path.EAR], "invalid Model"
        sig = -1 if model==pred_path.EAR else 1
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
                if sig in signals[i:j]:
                    latency.append(np.where(signals[i:j]==sig)[0][0])
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