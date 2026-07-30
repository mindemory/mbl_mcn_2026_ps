import numpy as np
import matplotlib.pyplot as plt

# Set dark background
plt.style.use('dark_background')

## 3.1
bin_width = 2 # ms
T = 100 # ms
tarray = np.arange(0, T, bin_width) # ms
stim = np.random.randn(tarray.shape[0]) 

## 3.2
omega = 0.3 # rad/ms
tau = 10 # ms
tarray_filter = np.arange(0, 50, bin_width)
kernel = np.exp(-tarray_filter/tau) * np.sin(omega * tarray_filter)



## Plotting
f, axs = plt.subplots(1, 2, figsize = (10, 5))
axs = axs.flatten()


axs[0].plot(tarray, stim, color = 'orange', linewidth = 1.2, marker = 's', markersize = 2)
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Stimulus intensity')
axs[0].set_title('Random stimulus')

axs[1].plot(tarray_filter, kernel, color = 'orange', linewidth = 1.2)
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Kernel value')
axs[1].set_title('Kernel')

plt.show()