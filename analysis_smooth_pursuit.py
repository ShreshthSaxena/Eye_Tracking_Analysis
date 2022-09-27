import os
import pandas as pd
import numpy as np
import ffmpeg
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy.stats.mstats import winsorize
from analysis_module import *
# import saccademodel
import ruptures
from scipy.stats import circmean as cim, circstd as cis
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter

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
        durations = {key:[] for key in self.angles}
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
            
            
            l = [(int(i),int(j)) for i,j in zip(time_to_frame(start_times, fps), time_to_frame(stop_times, fps))]
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
                sub = pred_df[pred_df.frame.between(pt[0],pt[1])] # movement duration (animation start (pt[0]) -> animation stop(pt[1]))
                trial_x[seq[index]].append(sub[colx])
                trial_y[seq[index]].append(sub[coly])
                try:
                    sub2 = pred_df[pred_df.frame.between(click_frames[index],pt[1])] # from user click to movement stop
                    win_len = len(sub2)//3 #ref for parameter determination https://doi.org/10.1016/j.rinp.2018.08.033
                    win_len = win_len+1 if win_len%2 == 0 else win_len
                    dist = (np.diff(apply_filter(sub2[colx], win_len=win_len),1)**2+np.diff(apply_filter(sub2[coly], win_len=win_len),1)**2)**(1/2)
    
                    algo = ruptures.Dynp(model="rbf", min_size=3, jump = 1).fit(dist)
                    result = algo.predict(n_bkps=2)
                    result = [r+1 for r in result] #correcting for the size reduction by 1 when diff is calculated
            
                    onset_time = frame_to_time(result[:1], fps)[0]
                    onset_time = click_times[index]+onset_time - start_times[index]
                    onsets[seq[index]].append(onset_time) # detected onset time from movement start in ms, rejected if onset_time<70ms
                    durations[seq[index]].append(frame_to_time([result[1]-result[0]], fps)[0]) #smooth pursuit duration
                    
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
        return trial_x,trial_y, onsets, durations
      
# Analysis functions

def apply_filter(data,win_len=15, moving_avg =False):
    
    if moving_avg:
        return data.rolling(win_len).mean().dropna()
    return savgol_filter(data, window_length = win_len, polyorder=1, deriv=0, mode='nearest', cval=0.0)

def process_trials(trial_x, trial_y, angles, show = False):
    avg = {k:[] for k in angles}
    for angle in trial_x.keys():
        for trial in range(10):

#             if trial_x[angle][trial].shape[0] < 15: #Handling with min trial sample = window_size
#                 continue
            #rolling mean
            try:
                win_len = len(trial_x[angle][trial])//2
                win_len = win_len+1 if win_len%2 == 0 else win_len
                sm_x = apply_filter(trial_x[angle][trial], win_len = win_len)
                sm_y = apply_filter(trial_y[angle][trial], win_len = win_len)
            except Exception as e:
                print(e)
                continue
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
    reg_angles = process_trials
    (trial_x,trial_y, angles)#, show=True)
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

#cmeans 
def get_win_sub(SP_cmeans, std_error = False):
    win_sub_cmean = {}
    win_sub_cse = {}
    win_sub_025 = {}
    win_sub_975 = {}
    for col in SP_cmeans.columns:
        win_means = circ_winsorize(SP_cmeans[col],col)
        win_sub_cmean[col] = cim(win_means, high=360, low=0) #winsorize circular mean
        if col==0 and win_sub_cmean[col]>180:
            win_sub_cmean[col] = win_sub_cmean[col] - 360
        win_sub_cse[col] = cis(win_means, high=360, low=0) #winsorize circular std
        if std_error:
            win_sub_cse[col] /= math.sqrt(SP_cmeans.shape[0]) #calculate std error instead

        win_means = np.array([x-360 if x-col > 180 else x for x in win_means])
        win_sub_025[col] = win_sub_cmean[col] - np.quantile(win_means, 0.025)
        win_sub_975[col] = np.quantile(win_means, 0.975) - win_sub_cmean[col]
    return win_sub_cmean, win_sub_cse, win_sub_025, win_sub_975


#Plotting Functions

def sp_plot(ax, angles, win_sub_cmean, win_sub_025, win_sub_975, color_line = "black", color_err = "teal"):
    
    ax.plot([0,330],[0,330],linestyle="--", lw=4, color = color_line)

    #Confidence interval for error bars
    ax.errorbar(angles, win_sub_cmean.values(), yerr = [list(win_sub_025.values()), list(win_sub_975.values())], ms = 10, linestyle = "None",elinewidth=4, marker="o", color = color_err)

    ax.set_xticks(angles)
    ax.set_yticks(angles)
    ax.tick_params(axis = 'both', labelsize=15)
#     ax.set_xlabel("target movement angle", fontsize=18)
#     ax.set_ylabel("mean predicted angle", fontsize=18)
    # plt.savefig("smooth_pursuit_faze.png")
    return ax

def sp_plot_single_trial(subb, block, trial, angle, colx = 'pred_x', coly = 'pred_y'):

        df = pd.read_csv(f"Subjects/{subb}/data.csv")
        task_df = df[df["Task_Name"] == "2. Smooth Pursuit"]
        row = task_df[(task_df["Trial_Nr"]==trial) & (task_df["Block_Nr"] == block)].iloc[0]
        palette = sns.color_palette('colorblind', 3)
        
        seq = [int(i) for i in row.angles.split(";")]
        index = seq.index(angle)
        
        fname = f"Subjects/{subb}/{row.rec_session_id}/blockNr_{row.Block_Nr}_taskNr_{row.Task_Nr}_trialNr_{row.Trial_Nr}_pursuit_rec.webm"
        c = get_frame_count(fname)
        vid_len = float(row.RecStop - row.RecStart)
        fps = c/vid_len #in ms

        start_times = row.anim_time.split("---values=")[1].strip("\"").split(";")
        start_times = [int(t) - int(row.RecStart) for t in start_times if len(t)>1]

        stop_times = row.ResetTimes.split("---values=")[1].strip("\"").split(";")
        stop_times = [int(t) - int(row.RecStart) for t in stop_times if len(t)>1]

        click_times = row.ClickTimes.split("---values=")[1].strip("\"").split(";")
        click_times = [int(t) - int(row.RecStart) for t in click_times if len(t)>1]
            
            
        l = [(int(i),int(j)) for i,j in zip(time_to_frame(start_times, fps), time_to_frame(stop_times, fps))]
        click_frames = time_to_frame(click_times, fps)

        try:
            print("Recording length (ffmpeg): ",ffmpeg.probe(fname)["format"]["duration"])
        except:
            pass
        print("file: ", fname)
        print("RecStop - RecStart : ",vid_len)
        print("Total Frame Count : ",c)
        print("used Frames Count : ",sum([pt[1]-pt[0] for pt in l]))
        print("FPS : ",fps*1000)
        print("start times : ",start_times)
        print("stop times : ",stop_times)
        print("click_times : ", click_times)
        print("diff : ", [int(i)-int(j) for i,j in zip(stop_times,start_times)])
        
        for i,model in enumerate([pred_path.MPII, pred_path.ETH, pred_path.FAZE]):
            print(model)
            pred_df = pd.read_csv(os.path.join(model.value, f"{subb}/pred_allcalib/Block_{row.Block_Nr}/Smooth Pursuit{row.Trial_Id}.csv"))
            sub = pred_df[pred_df.frame.between(l[index][0],l[index][1])] # movement duration (animation start (pt[0]) -> animation stop(pt[1]))

            sub2 = pred_df[pred_df.frame.between(click_frames[index],l[index][1])] # from user click to movement stop
            win_len = len(sub2)//3 #ref for parameter determination https://doi.org/10.1016/j.rinp.2018.08.033
            win_len = win_len+1 if win_len%2 == 0 else win_len
            dist = (np.diff(apply_filter(sub2[colx], win_len=win_len),1)**2+np.diff(apply_filter(sub2[coly], win_len=win_len),1)**2)**(1/2) #calculate derivative/vel after smoothing
            
            #Change point detection with dynamic programming, no. of breakpoints = 2 [onset and offset of SP]
            algo = ruptures.Dynp(model="rbf", min_size=3, jump = 1).fit(dist)
            result = algo.predict(n_bkps=2) 
            result = [r+1 for r in result] #correcting for diff reducing one sample
            print(result[:-1], f"no. of samples : {result[-1]}")
            

            sub2[colx].reset_index(drop=True).plot(figsize=(20,7), marker="o", color=palette[i])
            plt.axvline(result[0], linestyle = "--", color = palette[i])
            plt.axvline(result[1]+(0.08*(i-1)), linestyle = "-", color = palette[i], alpha=1)

        return 