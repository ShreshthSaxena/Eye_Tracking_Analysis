import os
import pandas as pd
import numpy as np
import ffmpeg    
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy.stats.mstats import winsorize
from sklearn import metrics
from analysis_module import *

class Zone_Classification():
    
    def __init__(self, subb, show=True):
        self.subb = subb
        self.df = pd.read_csv(os.path.join("Subjects",subb,"data.csv"))
        self.task_df = self.df[self.df["Task_Name"] == "4. Zone Classification"]
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
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/pred_allcalib/Block_{row.Block_Nr}/Zone Classification{row.Trial_Id}.csv"))
                
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
    
    
# subject wise confusion matrices
def classification_metrics(df, classification_report=False, grid_size=(4,4)):
    conf_matrices = []
       
    for index, row in df.iterrows():
        y_pred, y_true = [], []
        for pt in range(1,17):
            for trial in range(10):
                y_pred.append(get_zone(row.trial_x[pt][trial], row.trial_y[pt][trial], grid_size)+1)
                y_true.append(get_zone(*zone_center(pt), grid_size)+1)
        conf_matrix = metrics.confusion_matrix(y_true,y_pred)
        N = 16*10/(grid_size[0]*grid_size[1]) #number of trials for each zone
        conf_matrices.append(conf_matrix/N)
        if classification_report: #Update df
            report = metrics.classification_report(y_true,y_pred, output_dict = True)
            df.at[index, "accuracy"] = report["accuracy"]
            df.at[index, "precision"] = report["macro avg"]["precision"]
            df.at[index, "recall"] = report["macro avg"]["recall"]
            df.at[index, "f1-score"] = report["macro avg"]["f1-score"]
            df.at[index, "MCC"] = metrics.matthews_corrcoef(y_true, y_pred)
    return np.array(conf_matrices)
    

    
# previous method of calculating conf matric from mean of each subject
# def get_conf_matrix(df):
#     y_pred, y_true = [],[]
#     for i,row in df.iterrows():
#         for k in range(1,17):
#             y_pred.append(get_zone(row.mean_x[k], row.mean_y[k], (4,4))+1)
#             y_true.append(k)

#     conf_matrix = metrics.confusion_matrix(y_true,y_pred)
#     conf_matrix = (conf_matrix/df.shape[0])
#     return conf_matrix

# def classification_report(df, mcc=False):
#     y_pred, y_true = [],[]
#     for i,row in df.iterrows():
#         for k in range(1,17):
#             y_pred.append(get_zone(row.mean_x[k], row.mean_y[k], (4,4))+1)
#             y_true.append(k)
#     if mcc:
#         phi = metrics.matthews_corrcoef(y_true, y_pred)
#         return metrics.classification_report(y_true,y_pred), phi
#     else:
#         return metrics.classification_report(y_true,y_pred, output_dict = True)
    

def plot_zone(grid_size, ax, annotate=False, ann_fontsize=10, ann_list=None, **kwargs):
    '''
    plots a grid of provided size
    usage: plot_zone((4,4), axs)
    Parameters:
            ax : matplotlib axes 
            annotate (Bool): set True to display text in grid
            ann_list (list/array): array to use for displaying text. NOTE: check for size
            ann_fontsize (int): size of text (only used if annotate is True)
            **kwargs additional params for matplotlib grid
    '''
    x_ticks = [(i+1)*1600//grid_size[0] for i in range(grid_size[0])]
    y_ticks = [(i+1)*900//grid_size[1] for i in range(grid_size[1])]
    ax.set_xlim(0,1600)
    ax.set_ylim(0,900)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid("both", **kwargs)
    
    if annotate:
        pos = np.meshgrid(x_ticks, y_ticks)
        box_w = 1600//grid_size[0]
        box_h = 900//grid_size[1]
        for i,(x,y) in enumerate(zip(pos[0].flatten(),pos[1].flatten())):
            if ann_list is None:
                ax.text(x - (box_w/2), y - (box_h/2), str(i+1), fontsize= ann_fontsize, va='center', ha='center') #annotate with zone index
            else:
                ax.text(x - (box_w/2), y - (box_h/2), f"{round(ann_list[i],1)}%", fontsize= ann_fontsize, va='center', ha='center') #annonate with ann_list
    ax.invert_yaxis()