import os
import pandas as pd
import numpy as np
# import pylab
import ffmpeg
import matplotlib.pyplot as plt
import statistics
import ruptures
from scipy.stats.mstats import winsorize
from sklearn.cluster import AgglomerativeClustering
from analysis_module import *


class Blink_Detect():
    def __init__(self, subb, penalty_ruptures=0.8, show=True):
        '''
        set parameters:
        subb : Subject id
        penalty_ruptures : set penalty for ruptures changepoint detection algorith
        show : plot results (True/False)
        '''
        
        self.subb = subb
        self.df = pd.read_csv(f"Subjects/{subb}/data.csv")
        self.task_df = self.df[self.df["Task_Name"] == "5. Blink Detect"]
        self.penalty = penalty_ruptures
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
    
    def parse_trials(self, model, show = True, trial=None):
        '''
        returns trial_latencies, num_of_blinks, blink_durations
        '''
        #check if blink model is selected 
        assert model in [pred_path.RT_BENE, pred_path.EAR], "invalid Model"
        
        #ALGO1:  EAR outputs the eye area and RT-Bene outputs blink confidence which are complimentary in sign
        #sig = -1 if model==pred_path.EAR else 1
        trial_latencies = {key:[] for key in range(1,11)}
        num_of_blinks = {key:[] for key in range(1,11)}
        blink_durations = {key:[] for key in range(1,11)}
        
        trial_rows = self.task_df if trial==None else self.task_df[trial:trial+1]        
        for _,row in trial_rows.iterrows():
            rec_id = row.rec_session_id
            rec_name = f"blockNr_{row.Block_Nr}_taskNr_{row.Task_Nr}_trialNr_{row.Trial_Nr}_BlinkRec.webm"
            fname = f"Subjects/{self.subb}/{rec_id}/{rec_name}"

            start_times = row.beep_times.split("---values=")[1].strip("\"").split(";")
            start_times = [int(t) - int(row.RecStart) for t in start_times if len(t)>1]

            intervals = [int(i)-int(j) for i,j in zip(start_times[1:], start_times[:-1])]

            pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/{rec_name[:-5]}.csv"))

            #deal with missing data
            if (pred_df.iloc[:,-1].isnull().sum()) > (0.3*pred_df.shape[0]): 
                print("> 30% missing data")
                continue
            disp_df = pred_df.replace("NA",np.nan).interpolate(method='linear') #try other Interpolation methods?

            c = get_frame_count(fname)
            vid_len = float(row.RecStop - row.RecStart)
            fps = c/vid_len #in ms

            #ALGO1 : Robust Peak detection, Brakel et al.
        #     out = thresholding_algo(disp_df.iloc[:,-1].to_numpy(), 10, 4, 0.3)
        #     signals = np.pad(out["signals"][40:],(40,0),constant_values=0)
        #     onsets = time_to_frame(start_times, fps)
        #     assert len(signals) == disp_df.shape[0]

            #ALGO2 : Change point detection, can provide onset, offset, duration of blinks as well.

            #get changepoints
            algo = ruptures.Pelt(model="rbf", min_size=2, jump = 1).fit(disp_df.iloc[:,-1].to_numpy())
            result = algo.predict(pen=self.penalty) 

            #Filter out changepoints, criteria resp. to model chosen
            if model == pred_path.RT_BENE: 
                result = [r for r in result if (disp_df.iloc[r-3:r+3,-1].mean()>0.2)] #eliminate changepoints based on neighbouring peaks
                if len(result)<2: #check for clustering
                    continue
            else:
                result = [r for r in result if (disp_df.iloc[r-3:r+3,-1] < disp_df.iloc[:,-1].quantile(0.22)).any()] #eliminate changepoints on troughs 
                
            

            #Cluster onset,offset for each blink together
            blink_clus = []
            blink_dur = []
            cluster = AgglomerativeClustering(None, distance_threshold = 25).fit_predict(np.array(result).reshape(-1,1)) 
            for i in range(cluster.max()+1):
                c_i = np.where(cluster == i)[0]
                if c_i.shape[0]>=2:
                    on, off = result[c_i[0]], result[c_i[-1]]
                    blink_clus.append((on,off))      
                    blink_dur.append(frame_to_time(off-on, fps))

            #Calculate latency from beat onset
            latencies=[]
            mean_interval = time_to_frame([statistics.mean(intervals)],fps)[0]
            beat_onsets = time_to_frame(start_times, fps)
            blink_onsets = sorted([i[0] for i in blink_clus])
            trial_blinks = [b for b in blink_clus if beat_onsets[1]< b[0]< beat_onsets[-1]+mean_interval]
            for on,off in zip(beat_onsets, beat_onsets[1:]+[beat_onsets[-1]+mean_interval]):
                sub_blinks = [ i for i in blink_onsets if on<i<off ]
                if len(sub_blinks)>0:
                    latencies.append(frame_to_time(sub_blinks[0]-on, fps))
                else:
                    latencies.append(np.nan)

            #Logging (update when changing ALGO"
            if show:
                print("Recording Name : ",rec_name)
                print("Recording length : ",ffmpeg.probe(fname)["format"]["duration"])
                print("RecStop - RecStart : ",vid_len)
                print("Frame Count : ",c)
                print("FPS : ",fps*1000)
                print("start times : ",start_times)
                print("inter-beat duration : ", intervals, "Avg: ", statistics.mean(intervals)) 
                print("blinks : ", blink_clus)
                print("No. of blinks : ", len(blink_clus))
                print("Latencies: ",latencies)
                print("Blink durations : ", blink_durations, "Avg: ", statistics.mean(blink_dur)) 
                disp_df.plot(x="frame", y=disp_df.columns[-1], figsize=(20,6), color="#F0BC42" if model.name=="RT_BENE" else "#8E1F2F", linewidth = 2.5)
                ymin,ymax = plt.gca().get_ylim()
                plt.vlines(time_to_frame(start_times, fps),ymin,ymax, linestyles ='dashed', colors = 'gray', linewidth=2.5)
#                 plt.vlines(result, ymin, ymax, colors="k", alpha = 0.4)
                plt.vlines([b[0] for b in blink_clus],ymin,ymax, linestyles ='-', colors = '#696969', linewidth=2.5)
                plt.vlines([b[1] for b in blink_clus],ymin,ymax, linestyles ='-', colors = 'black', linewidth=2.5)
                plt.legend(["model output", "beat onset","predicted blink onset","predicted blink offset"], loc="upper right" if model.name=="RT_BENE" else "lower right", fontsize=11)#, bbox_to_anchor=(1.15, 1))
                plt.xlabel("Frame", fontsize = 23, fontweight=500)
                plt.ylabel("Blink probability" if model.name == "RT_BENE" else "EAR: Eye Aspect Ratio", fontsize=20, fontweight = 500)
                plt.tick_params(axis='both', which='major', labelsize=20, width = 10)
                plt.savefig(f"full_paper_plots/{model.name}_trial.png", transparent = True, dpi = 300)
                plt.show()
                
            #     plt.figure(figsize=(20,6))
            #     plt.plot(signals)
            #     plt.show()
                print("-"*50)
            trial_latencies[row.Trial_Id + (5 if row.Block_Nr == 13 else 0)] = np.array(latencies)
            num_of_blinks[row.Trial_Id + (5 if row.Block_Nr == 13 else 0)] = len(trial_blinks)
            blink_durations[row.Trial_Id + (5 if row.Block_Nr == 13 else 0)] = np.array(blink_dur)
        return trial_latencies, num_of_blinks, blink_durations
    
    
#Brakel, J.P.G. van (2014). "Robust peak detection algorithm using z-scores". Stack Overflow. Available at: https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362 (version: 2020-11-08).
# def thresholding_algo(y, lag, threshold, influence):
#     signals = np.zeros(len(y))
#     filteredY = np.array(y)
#     avgFilter = [0]*len(y)
#     stdFilter = [0]*len(y)
#     avgFilter[lag - 1] = np.mean(y[0:lag])
#     stdFilter[lag - 1] = np.std(y[0:lag])
#     for i in range(lag, len(y)):
#         if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
#             if y[i] > avgFilter[i-1]:
#                 signals[i] = 1
#             else:
#                 signals[i] = -1

#             filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
#             avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
#             stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
#         else:
#             signals[i] = 0
#             filteredY[i] = y[i]
#             avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
#             stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

#     return dict(signals = np.asarray(signals),
#                 avgFilter = np.asarray(avgFilter),
#                 stdFilter = np.asarray(stdFilter))