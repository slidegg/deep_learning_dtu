# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:19:39 2021

@author: xysti
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('log.csv')
df.columns
df1 = df.iloc[:, :]


# Plot every 5th line mean episode rewards&episode length
ax = plt.gca()
ax2 = plt.gca()

df1.plot(kind='line',x='num_episodes',y='mean_episode_rewards', color='blue',ax=ax)
# df2.plot(kind='line',x='num_episodes',y='mean_episode_rewards', color='red',ax=ax2)

plt.savefig('DL_results_mean_episode_rewards_new.png')
plt.cla();


# plot loss_pi_loss
ax = plt.gca()
df1.plot(kind='line',x='num_episodes',y='loss_pi_loss', ylabel = 'Pi Loss', title = 'Policy Loss', label = 'pi_loss', ax=ax)
plt.savefig('DL_pi_loss_new.png')


# plot loss_v_loss
ax = plt.gca()
df1.plot(kind='line',x='num_episodes',y='loss_v_loss', ylabel = 'Value Loss', title = 'Value Loss', label = 'value_loss', ax=ax)
plt.savefig('DL_value_loss_new.png')


# plot loss entropy
ax = plt.gca()
df1.plot(kind='line',x='num_episodes',y='loss_entropy_loss', ylabel = 'Entropy Loss', title = 'Entropy Loss', label = 'entropy_loss', ax=ax)
plt.savefig('DL_entropy_loss_new.png')