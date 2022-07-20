import os
import pandas as pd
import numpy as np
import ffmpeg    
import math, statistics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from analysis_module import *

class Fixation():
    def __init__(self, subb, show=True):
        
        self.gt_points = [(16, 16), (16, 450), (16, 884), (539, 305), (539, 595), (800, 16), (800, 884), (1061, 305), (1061, 595), (1584, 16), (1584, 450), (1584, 884), (800, 450)]
        self.subb = subb
        self.df = pd.read_csv(f"Subjects/{subb}/data.csv")
        self.task_df = self.df[self.df["Task_Name"] == "1. Fixation"]
        self.palette = sns.color_palette('colorblind',14)
        if show:
            print("task_df summary \n {}".format(self.summary()))

    def summary(self):
        print(self.task_df.iloc[0].dropna()[8:17])
    
    def parse_trials(self, model, col_x, col_y, pred_final = False, pred_allcalib=False, calib_test=None, model_outputs = False, model_outputs_FT=False, eth_kalman = False, show = True, df_path=None):
        """
        
        pick appropriate column names for each mode using col_x, col_y
        choose a mode from 
        
        model_outputs: use uncalibrated predictions
        
        pred_final: use Beg+Mid+End calibrated predictions (only E calib)
        
        pred_allcalib: use Beg+Mid+End with E+SP calib predictions (best resulting strategy ETRA shortpaper)
        
        eth_kalman: use kalman filtered predictions from ETH !!make sure that model==pred_path.ETH!! 
        
        model_outputs_FT: use fine tuned predictions !!make sure that model==pred_path.FAZE!!
        
        calib_test: use for calibration tests 1,2,3
        
        """
        assert pred_final+pred_allcalib+(calib_test!=None)+model_outputs+(model==pred_path.FAZE and model_outputs_FT)+(model==pred_path.ETH and eth_kalman) == 1, "Check function arguments"
        
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
                if df_path != None:
                    pred_df = pd.read_csv(os.path.join(model.value, self.subb,df_path,f"Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
                else:
                    pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/model_outputs/Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))                
            elif calib_test != None:
                if df_path == None:
                    pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/calib_test{calib_test}/outputs/Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
                else: pred_df = pd.read_csv(os.path.join(model.value, self.subb, df_path, f"Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
            elif pred_final:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/pred_final/Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
            elif pred_allcalib:
#                 dir_ = "pred_FT_allcalib" if model == pred_path.FAZE else "pred_allcalib" 
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/pred_allcalib/Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
            elif model == pred_path.ETH and eth_kalman == True:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/kalman_preds/Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
            elif model == pred_path.FAZE and model_outputs_FT == True:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/model_outputs_FT/Block_{row.Block_Nr}/Fixation{row.Trial_Id}.csv"))
            else:
                continue
                
            if model==pred_path.MPII and pred_df.frame.isna().any():
                pred_df["frame"] = pred_df.index+1 #correcting MPII csv error
            
            for index,pt in enumerate(l):
                sub = pred_df[pred_df.frame.between(pt[0],pt[1])]
                
                # Ehinger et al uses fixation detection algo. to identify last fixation, we're using medians of all points
#                 X = winsorize(sub.poly_x, limits=[0.1,0.1])
#                 Y = winsorize(sub.poly_y, limits=[0.1,0.1])
                try:
                    #accuracy
                    trial_x[fix_seq[index]].append(round(statistics.median(sub[col_x]),2))
                    trial_y[fix_seq[index]].append(round(statistics.median(sub[col_y]),2))
                except Exception as e:
                    print(e)
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
    
## Task Measures
def get_fix_acc(GT, trial_x,trial_y, pt_index = range(1,14)):
    error_trials = []
    for trial in range(10):
        error_pts = []
        #winsorize means over points
        for pt in pt_index:
            error_pts.append(get_dist(GT[pt-1], (trial_x[pt][trial],trial_y[pt][trial])))
#         assert len(error_pts)==13, print(len(error_pts))
        error_trials.append(winsorize(pd.Series(error_pts).dropna(), limits=[0.1,0.1]).mean())
    #finally winsorize mean over trials 
    acc = winsorize(error_trials, limits=[0.1,0.1]).mean() 
    return acc

def get_fix_precision(rms, std, pt_index = range(1,14)):
    np_rms = rms.copy()
    np_std = std.copy()
    pt_index = [i-1 for i in pt_index] #index starting from 0
    #winsorize mean over points
    for i in range(10):
        np_rms[pt_index,i] = winsorize(np_rms[pt_index,i], limits=[0.1,0.1])
        np_std[pt_index,i] = winsorize(np_std[pt_index,i], limits=[0.1,0.1])
    #mean over trials
    mean_rms = winsorize(np_rms[pt_index, :].mean(axis=0), limits=[0.1,0.1]).mean()
    mean_std = winsorize(np_std[pt_index, :].mean(axis=0), limits=[0.1,0.1]).mean()
    return mean_rms, mean_std

    
#Plotting functions
def confidence_ellipse(x, y, ax, n_std=3, facecolor='none', **kwargs):

    assert x.size == y.size, "x and y must be the same size"

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 4, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def fix_plot(pts, df, ax,kind = 'def', xmin=0, xmax=1600, ymin=0,ymax=900, color = "teal"):
    '''
    use kind from one of the following: all_subjects, err, ellipse
    '''
#     palette = iter(sns.color_palette('husl',13))
    for key in range(13):
        gt = pts[key]
        xs = winsorize(df['mean_x'].apply( lambda x: x[key+1] ), limits=[0.1,0.1]) 
        ys = winsorize(df['mean_y'].apply( lambda y: y[key+1] ), limits=[0.1,0.1])
        if kind == 'all_subjects':
            ax.plot(gt[0],gt[1],marker="+", mew=3, markersize = 12, color = 'k')
            ax.scatter(xs,ys, color = color, s=60)
        
        elif kind == 'err':
            ax.errorbar(xs.mean(),ys.mean(),xerr = xs.std(),yerr = ys.std(),color = color,alpha=0.8, linestyle="None", marker = ".")
            ax.plot(gt[0],gt[1],marker="h", markersize = 10, color = "k")
        
        elif kind == 'ellipse':
            ax.scatter(xs.mean(), ys.mean(), s=50, color = color)
            ax.plot(gt[0],gt[1],marker="+", mew=3, markersize = 12, color = 'k')
#             ax.errorbar(xs.mean(),ys.mean(),xerr = xs.std(),yerr = ys.std(),color = c,alpha=0.3, linestyle="None", marker = ".")
            ellipse = Ellipse((xs.mean(), ys.mean()), width=xs.std()*2, height=ys.std()*2,
                      facecolor=color, edgecolor=color, alpha=0.2, lw=5)
            ax.add_patch(ellipse)
#             ax.legend(["+",".","--"], )
        
        elif kind == 'cov_ellipse':
            ax.scatter(xs.mean(), ys.mean(), s=50, color = color)
            ax.plot(gt[0],gt[1],marker="+", mew=5, markersize = 12, color = "k")
            confidence_ellipse(xs, ys, ax, n_std=1, edgecolor="black", alpha=0.8 )#, linestyle="--")
        
        elif kind == 'def':
            ax.plot(xs.mean(),ys.mean(),marker = ".", markersize = 30 , color = color)
            ax.plot(gt[0],gt[1],marker="+", markersize = 15, color = "k")  

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xticks([0,400,800,1200,1600])
    ax.set_yticks([0,300,600,900])
    return


