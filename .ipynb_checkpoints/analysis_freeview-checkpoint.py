import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy.stats.mstats import winsorize
from analysis_module import *

# import sys
# sys.path.append('Models/blink_Soukuova_and_Check/')
# from quick_blinks import get_ear
from st_dbscan import ST_DBSCAN #using event classification instead of blink det

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
                
            if model_outputs: #if uncalibrated data needs to be analysed
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/model_outputs/Block_{row.Block_Nr}/FreeView{row.Trial_Id}.csv"))
            else:
                pred_df = pd.read_csv(os.path.join(model.value, f"{self.subb}/pred_allcalib/Block_{row.Block_Nr}/FreeView{row.Trial_Id}.csv"))
                                    
            if show:
                ax = plot_one_subject(row["Trial_Id"],pred_df["pred_x"],pred_df["pred_y"])
                plt.gca().invert_yaxis()
                ax.plot()
                plt.show()
            trial_x[row["Trial_Id"]] = pred_df["pred_x"]
            trial_y[row["Trial_Id"]] = pred_df["pred_y"]
        return trial_x, trial_y    
    
#ST-DBSCAN centroids
def DB_centroids(data, model):
    eps = {pred_path.MPII: 61, pred_path.ETH: 48, pred_path.FAZE: 14} #eps1 = precision RMS/2 for each model
    
    st_dbscan = ST_DBSCAN(eps1 = eps[model], eps2 = 6, min_samples = 2).fit(data) #eps2 = 6
    centroids = np.empty((0,2), int)
    #Exclude outliers and first fixation
    for cl in range(1, max(st_dbscan.labels)):
#         print(np.mean(data[st_dbscan.labels==cl,:], axis = 0)[1:])
        centroids = np.append(centroids, np.median(data[st_dbscan.labels==cl,:], axis = 0)[1:].reshape(-1,2), axis=0)
    return centroids.astype(int)
    
## Plotting functions
  
def plot_one_subject(trial, trial_x, trial_y, xlim=1600 ,ylim=900  , figsize=(18,9), n=14, text=False, ax=None, **kwargs):
        img_name = FV_IMAGES[trial]
        if ax==None:
            fig,ax = plt.subplots(figsize=figsize)
        img = cv2.imread("FreeView_Images/"+img_name+".jpeg")
        print("Trial_Id", trial, "image name", img_name)
        img,xbounds, ybounds = labVanced_present(img)
        ax.imshow(imutils.opencv2matplotlib(img), origin="lower", extent = [xbounds[0],xbounds[1],ybounds[0], ybounds[1]])
        ax.plot(trial_x,trial_y, **kwargs)
        if text:
            for i, x, y in zip(range(len(trial_x)), trial_x, trial_y):
                ax.text(x, y, str(i), color="k", fontsize=17, ma='center', va='center', ha='center')
#         ax.set_xlim(0,xlim)
#         ax.set_ylim(0,ylim)
        return ax
    
def plot_all_subjects(df, model, xlim=1600 ,ylim=900  , figsize=(18,9)):
    x_pts = df.trial_x.apply(pd.Series) #rows: Subjects, columns: trials/images 
    y_pts = df.trial_y.apply(pd.Series)
         
    for trial in x_pts.columns[:]: # images
        #plot image
        img_name = FV_IMAGES[trial]
        plt.figure(figsize=(16,6))
        img = cv2.imread("FreeView_Images/"+img_name+".jpeg")
        print("Trial_Id", trial, "image name", img_name)
        img,xbounds, ybounds = labVanced_present(img)
        plt.imshow(imutils.opencv2matplotlib(img), origin="lower", extent = [xbounds[0],xbounds[1],ybounds[0], ybounds[1]])
        
        #plot gaze pts for each subject
        for subject in x_pts.index:
            data = (pd.concat([x_pts[trial][subject], y_pts[trial][subject]], axis = 1).reset_index().values)
            centroids = DB_centroids(data, model)

            try:
                plt.scatter(x=centroids[:,0], y=centroids[:,1], color="orange", alpha=0.4)
            except IndexError as e:
                print(f"subject {subject}, centroids: {len(centroids)}, exception: {e}")        
        plt.gca().invert_yaxis()
        plt.show()
    
def stationary_entropy(data, bin_size=54, screen_dim=(1600,900), show = False):
    '''
    Parameters:
        data - Numpy array of coordinates (x,y) with shape (N,2) where N=number of gaze samples
        bin_size - size of histogram bins, default set to 1 visual degree for our study
        screen_dim - (width, height) of screen
        show - set True to print entropy
    '''
    df = pd.DataFrame(data, columns=('x','y'))
    df['x_range'] = pd.cut(df.x, np.arange(0, screen_dim[0], bin_size), right=False)
    df['y_range'] = pd.cut(df.y, np.arange(0, screen_dim[1], bin_size), right=False)
    df=df.groupby(['x_range','y_range']).size().reset_index().rename(columns={0:'count'})
    df['p']=df['count']/df['count'].sum()
    df['p*log(p)']= np.log2(df['p'])*df['p']
    max_H = math.log2((screen_dim[0]/bin_size)*(screen_dim[1]/bin_size))
    H = abs(df['p*log(p)'].sum())
    norm_H = H/max_H
    if show:
        print('State Spaces',screen_dim[0]/bin_size, '*', screen_dim[1]/bin_size, '=', (screen_dim[0]/bin_size)*(screen_dim[1]/bin_size))
        print('Maximum entropy', max_H)
        print('Observed entropy' , H)
        print('Normalised entropy', norm_H)
    return norm_H
    
## Analysis functions and metrics
# Ref:
# https://github.com/herrlich10/saliency/blob/master/benchmark/utils.py
# https://github.com/cvzoya/saliency/tree/master/code_forMetrics

def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.
    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.
    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res

def CC(saliency_map1, saliency_map2):
    '''
    Pearson's correlation coefficient between two different saliency maps
    (CC=0 for uncorrelated maps, CC=1 for perfect linear correlation).

    '''
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    assert map1.shape == map2.shape, "Size of two maps do not match"
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0,1]

def SIM(saliency_map1, saliency_map2):
    '''
    Similarity between two different saliency maps when viewed as distributions
    (SIM=1 means the distributions are identical).
    This similarity measure is also called **histogram intersection**.
    '''
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    assert map1.shape == map2.shape, "Size of two maps do not match"
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)
 
def AUC_Judd(gt_map, pred_indices):
    ##normalize
    gt_map = (gt_map-gt_map.min())/(gt_map.max() - gt_map.min()) 

    #from MIT saliency metrics Matlab/py (/herrlich10/saliency) implementation
    S = gt_map.ravel()
    F = pred_indices.ravel().astype(bool)
    S_fix = S[F]
    n_fix = len(S_fix)
    n_pixels = len(S)

    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp)

def std_2D(x,y):
    x_mean = int(statistics.mean(x))
    y_mean  = int(statistics.mean(y))
    d_mean=[]
    for i,j in zip(x,y):
        d_mean.append(distance.euclidean([x_mean,y_mean],[i,j]))
    return statistics.stdev(d_mean)