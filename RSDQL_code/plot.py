
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

file_path1 = 'trainloss.txt'
file_path2 = 'reward.txt'

data_loss = pd.read_csv(file_path1, sep=',', header=None,names=['epsiodes','loss'])
data_loss = data_loss.values

reward = pd.read_csv(file_path2, sep=',', header=None,names=['epsiodes','total_reward'])
reward = reward.values

x = data_loss[:,0]
y = data_loss[:,1]

m = reward[:,0]
n = reward[:,1]

fig1 = plt.figure(figsize = (7,5))  
pl.plot(x,y,'g-',label=u'loss')
pl.xlabel(u'episodes')
pl.ylabel(u'loss')
#pl.show()

fig2 = plt.figure(figsize = (7,5))  
pl.plot(m,n,'g-',label=u'reward')
pl.xlabel(u'episodes')
pl.ylabel(u'reward')
pl.show()