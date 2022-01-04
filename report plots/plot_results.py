# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:19:39 2021

@author: xysti
"""

import pandas as pd
import matplotlib.pyplot as plt


# log test: betas(0.95, 0.999), value_coeff = 0.7, eps = 1e-4


#df = pd.read_csv('log_latest3.csv') # log_latest    log2
df2 = pd.read_csv('log_test.csv')
df = pd.read_csv('log1.csv')
df3 = pd.read_csv('log_old.csv')

df.columns
#df1 = df.iloc[::5, :]  # every 5th line


# plot rewards and episode length
ax = plt.gca()
df.plot(kind='line',x='num_episodes',y='mean_episode_len', ylabel = 'Time alive (counted in steps)', title = 'Time that the episode lasted', xlabel = 'Episode', label = 'time alive', color='red', ax=ax)

ax = plt.gca()
df.plot(kind='line',x='num_episodes',y='mean_episode_rewards', ylabel = 'Rewards (kills)', title = 'Agent rewards', xlabel = 'Episode', label = 'episode rewards', ax=ax)



# plot loss_pi_loss
ax = plt.gca()
df.plot(kind='line',x='num_episodes',y='loss_pi_loss', ylabel = 'Pi Loss', title = 'Policy Loss', label = 'pi loss', ax=ax)
df2.plot(kind='line',x='num_episodes',y='loss_pi_loss', ylabel = 'Pi Loss', title = 'Policy Loss', label = 'pi loss test', color = 'red', ax=ax)
df3.plot(kind='line',x='num_episodes',y='loss_pi_loss', ylabel = 'Pi Loss', title = 'Policy Loss', label = 'pi loss test default', color = 'orange', ax=ax)


# plot loss_v_loss
ax = plt.gca()
df.plot(kind='line',x='num_episodes',y='loss_v_loss', ylabel = 'Value Loss', title = 'Value Loss', label = 'value loss', ax=ax)
df2.plot(kind='line',x='num_episodes',y='loss_v_loss', ylabel = 'Value Loss', title = 'Value Loss', label = 'value loss test', color = 'red', ax=ax)
df3.plot(kind='line',x='num_episodes',y='loss_v_loss', ylabel = 'Value Loss', title = 'Value Loss', label = 'value loss default', color = 'orange', ax=ax)


# plot loss entropy
ax = plt.gca()
df.plot(kind='line',x='num_episodes',y='loss_entropy_loss', ylabel = 'Entropy Loss', title = 'Entropy Loss', label = 'entropy loss', ax=ax)
df2.plot(kind='line',x='num_episodes',y='loss_entropy_loss', ylabel = 'Entropy Loss', title = 'Entropy Loss', label = 'entropy loss test', color = 'red', ax=ax)
df3.plot(kind='line',x='num_episodes',y='loss_entropy_loss', ylabel = 'Entropy Loss', title = 'Entropy Loss', label = 'entropy loss default', color = 'orange', ax=ax)














