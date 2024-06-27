import numpy as np
import matplotlib.pyplot as plt
import onnx

ctrl_arr = np.load("exp_dir/ctrl_epoch_losses.npy")
# print(ctrl_arr)

# vae_arr = np.loadtxt("exp_dir/vae_epoch_losses.txt")

plt.plot(ctrl_arr)
plt.title("VAE loss")
plt.show()