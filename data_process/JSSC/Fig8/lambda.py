import cv2
import os
import torch 
import numpy as np
import sys
import matplotlib.pyplot as plt

lambda_i = [0, 0.5, 2, 4, 8, 20]
sparsity = [50, 60, 70, 80, 90, 95]
accuracy = [0.973, 0.968, 0.965, 0.964, 0.962, 0.909]

fig, ax = plt.subplots(1,1)
color = (0, 0, 0)
ax.plot(lambda_i, sparsity, 'x-', marker='s', color=color,  linewidth=2, markersize=8, label='Sparsity')
ax.set_xlabel(r"$\bf{Lambda}$"+' ' + r"$\bf{}$", size=10, family="Arial")
ax.set_ylabel(r"$\bf{Sparsity}$"+' ' + r"$\bf{(%)}$", size=10, color=color, family="Arial")
ax.legend (loc='upper left')
plt.grid(True)
from matplotlib.pyplot import MultipleLocator
inst = plt.gca()
x_major_locator=MultipleLocator(0.1)
inst.xaxis.set_major_locator(x_major_locator)
inst.set_xlim(0.65,1.25)


ax1 = ax.twinx()
color1 = (1, 0, 0)
ax1.plot(lambda_i, accuracy, '--', marker='^', color=color1,  linewidth=2, markersize=9, label='Accuracy')
ax1.set_ylabel(r"$\bf{Accuracy}$"+' ' + r"$\bf{}$", size=10, color=color1, family="Arial")
# ax1.legend(loc='upper right')
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax1.get_legend_handles_labels()
plt.legend(handles1+handles2, labels1+labels2, loc='upper left')

ax.spines['left'].set_color(color)
ax.tick_params(axis='y', colors=color)
# ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax1.spines['right'].set_color(color1)
ax1.tick_params(axis='y', colors=color1)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)

plt.savefig(os.path.join(sys.path[0],'lambda.svg'), format='svg')
plt.show()