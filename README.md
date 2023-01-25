# LiDAR Aided Future Beam Prediction in Real-World Millimeter Wave V2I Communications
This is a python code package related to the following article:
S. Jiang, G. Charan and A. Alkhateeb, "[LiDAR Aided Future Beam Prediction in Real-World Millimeter Wave V2I Communications](https://ieeexplore.ieee.org/document/9939167)," in IEEE Wireless Communications Letters, 2022.

# Instructions to Reproduce the Results 
The scripts for generating the results of the ML solutions in the paper. This script adopts Scenario 8 of DeepSense6G dataset.

**To reproduce the results, please follow these steps:**
1. Download [DeepSense 6G/Scenario 8](https://deepsense6g.net/scenario-8/).
2. Download (or clone) the repository into a directory.
3. Extract the dataset into the repository directory.
4. Run train_model.py file.

**Results of the script**
| Solution       | Current beam | Future beam 1 | Future beam 2 | Future beam 3 |
| :------------- | ------------ | ------------- | ------------- | ------------- |
| Top-1          |     58.4%    |     54.9%     |     51.1%     |     45.8%     |
| Top-5          |     94.9%    |     94.3%     |     93.4%     |     92.5%     |

If you have any questions regarding the code and used dataset, please write to DeepSense 6G dataset forum https://deepsense6g.net/forum/ or contact [Shuaifeng Jiang](mailto:sjiang74@asu.edu).

# Abstract of the Article
This paper presents the first large-scale real-world evaluation for using LiDAR data to guide the mmWave beam prediction task. A machine learning (ML) model that leverages LiDAR sensory data to predict the current and future beams was developed. Based on the large-scale real-world dataset, DeepSense 6G, this model was evaluated in a vehicle-to-infrastructure communication scenario with highly-mobile vehicles. The experimental results show that the developed LiDAR-aided beam prediction and tracking model can predict the optimal beam in 95% of the cases and with around 90% reduction in the beam training overhead. The LiDAR-aided beam tracking achieves comparable accuracy performance to a baseline solution that has perfect knowledge of the previous optimal beams, without requiring any knowledge about the previous optimal beam information and without any need for beam calibration. This highlights a promising solution for the critical beam alignment challenges in mmWave and terahertz communication systems.
# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
If you in any way use this code for research that results in publications, please cite our original article:
> S. Jiang, G. Charan and A. Alkhateeb, "LiDAR Aided Future Beam Prediction in Real-World Millimeter Wave V2I Communications," in IEEE Wireless Communications Letters, 2022, doi: 10.1109/LWC.2022.3219409.

If you use the [DeepSense 6G dataset](www.deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” to be available on arXiv, 2022. [Online]. Available: https://www.DeepSense6G.net
