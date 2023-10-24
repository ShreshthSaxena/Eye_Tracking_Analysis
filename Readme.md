# Deep learning models for webcam eye-tracking in online experiments


This repository contains analysis scripts that can be used to generate/replicate the plots and results reported in our paper.

**Title:** Deep learning models for webcam eye tracking in online experiments\
**Published In:** Behavior Research Methods\
**DOI:** https://doi.org/10.3758/s13428-023-02190-6\

## Overview

We conducted an online study to evaluate the performance of deep learning models when applied for webcam-based eye-tracking in remote settings. The models were evaluated on a custom eye-tracking task battery[^1] [^9] that tests performance over a multitude of gaze and blink measures.

## Study Design

![](task_seq.png)


For further details and results of our study please refer to our publications [^1][^2]. \
The online experiment performed by the participants can be accessed here: https://www.labvanced.com/player.html?id=41124 \
The OSF project linked with this study is available here: https://osf.io/qh8kx/


<!-- ## Abstract -->


## Code Implementation

> Gaze and blink predictions were evaluated using the following models.
> - MPIIGaze: https://github.com/hysts/pytorch_mpiigaze [^3]  
> - ETHXGaze:https://github.com/xucong-zhang/ETH-XGaze [^4]
> - FAZE: https://github.com/NVlabs/few_shot_gaze [^5]
> - RT_BENE: https://github.com/Tobias-Fischer/rt_gene [^6]

All source code used to analyse the predictions and generate figures in the paper are in this repository. 

Analysis and screening of participant data (See OSF pre-registration) is run using [Participant_Analysis.ipynb](Task_Analysis.ipynb)

Calibration analysis and comparison of different strategies for the ETRA paper (Saxena et al. 2022 [^2]) is run using [Calib_tests.ipynb](Calib_tests.ipynb)

Task-wise parsing of eye-tracking data and utility functions are provided in analysis scripts named following the convention "analysis_[task].py"

Task-wise analyses are all run inside [Task_Analysis.ipynb](Task_Analysis.ipynb).

Anova tests on individual task performance and the 3x4 ANOVA across all 3 models and 4 tasks are run inside [Task_ANOVA.ipynb](Task_ANOVA.ipynb).

Saved data frames and example data used in these files are provided in `csv_backup`

## Dependencies

Use the requirements.txt file to install all dependencies. The code was developed and run using python version 3.7.0

```
pip install -r requirements.txt
```


## License

We make all our work available under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


## References

[^1]: Saxena, S., Fink, L.K. & Lange, E.B. Deep learning models for webcam eye tracking in online experiments. Behav Res (2023). https://doi.org/10.3758/s13428-023-02190-6

[^2]: Shreshth Saxena, Elke Lange, and Lauren Fink. 2022. Towards efficient calibration for webcam eye-tracking in online experiments. In 2022 Symposium on Eye Tracking Research and Applications (ETRA '22). Association for Computing Machinery, New York, NY, USA, Article 27, 1–7. https://doi.org/10.1145/3517031.3529645

[^3]: Xucong Zhang, Yusuke Sugano, Mario Fritz and Andreas Bulling. 2015. Appearance-based gaze estimation in the wild. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 4511–4520. https://doi.org/10.1109/CVPR.2015.7299081.Google ScholarCross Ref

[^4]: Xucong Zhang, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang and Otmar Hilliges. 2020. ETH-XGaze: A large scale dataset for gaze estimation under extreme head pose and gaze variation. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 12350 LNCS, 365–381. https://doi.org/10.1007/978-3-030-58558-7_22.Google Scholar

[^5]: Seonwook Park, Shalini De Mello, Pavlo Molchanov, Umar Iqbal, Otmar Hilliges and Jan Kautz. 2019. Few-shot adaptive gaze estimation. Proceedings of the IEEE International Conference on Computer Vision, 9367–9376. https://doi.org/10.1109/ICCV.2019.00946.Google ScholarCross Ref

[^6]: Cortacero, K., Fischer, T., & Demiris, Y. (n.d.). RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments. Retrieved October 8, 2021, from www.imperial.ac.uk/Personal-Robotics/

[^7]: Soukupová, T., & Cech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks.In 21st Computer Vision Winter Workshop, Rimske Toplice, Slovenia.

[^8]: Judd, T., Ehinger, K., Durand, F., & Torralba, A. (n.d.). Learning to Predict Where Humans Look. Proceedings of the IEEE 12th International Conference on Computer Vision, 2106–2113. https://doi.org/10.1109/ICCV.2009.5459462.

[^9]: Ehinger, B. V., Groß, K., Ibs, I., & König, P. (2019). A new comprehensive eye-tracking test battery concurrently evaluating the Pupil Labs glasses and the EyeLink 1000. PeerJ, 2019(7), 1–43. https://doi.org/10.7717/peerj.7086

[^10]: Truong, C., Oudre, L., & Vayatis, N. (2018). ruptures: Change point detection in Python (arXiv:1801.00826). arXiv. https://doi.org/10.48550/arXiv.1801.00826

[^11]: Cakmak, E., Plank, M., Calovi, D. S., Jordan, A., & Keim, D. (2021). Spatio-temporal clustering benchmark for collective animal behavior. Proceedings of the 1st ACM SIGSPATIAL International Workshop on Animal Movement Ecology and Human Mobility, 5–8. https://doi.org/10.1145/3486637.3489487



