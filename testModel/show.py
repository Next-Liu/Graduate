import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 指定.npy文件路径
file_path1 = "../results/informer_custom_ftMS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/real_prediction.npy"
file_path2 = "../results/informer_custom_ftMS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/pred.npy"
file_path3 = "../results/informer_custom_ftMS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0/true.npy"

# 使用NumPy加载.npy文件
true_value = []
pred_value = []

real_prediction = np.load(file_path1)
prediction = np.load(file_path2)
true = np.load(file_path3)
print(real_prediction.shape)

# draw prediction
plt.figure()
plt.plot(true[0, :, -1], label='true')
plt.plot(real_prediction[0, :, -1], label='real_prediction')

plt.legend()
plt.show()
