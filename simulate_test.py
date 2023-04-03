# Author: Siddhant Tandon
# Description: Testing for generating RGB images for robot simulation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
 
fig = plt.figure(figsize=(10, 10), dpi = 100)
x = [2.5]
y = [2.5]
ln, = plt.plot(x, y, marker="o", ms=50, mfc='k')
# ln.set(gapcolor='black')
plt.axis([0, 10, 0, 10])
# plt.axis('off')
plt.grid()
path = '/usr/bin/ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = path
def update(frame):
    if x[-1] ==2.5 and y[-1] < 6.5:
        x.append(x[-1])
        y.append(y[-1] + 1)
    elif y[-1] == 6.5 and x[-1] < 6.5:
        x.append(x[-1] + 1)
        y.append(y[-1])
    elif x[-1] ==6.5 and y[-1] <= 6.5 and y[-1] >2.5:
        x.append(x[-1])
        y.append(y[-1] - 1)
    elif x[-1] >2.5 and x[-1] <=6.5 and y[-1] ==2.5:
        x.append(x[-1] - 1)
        y.append(y[-1])        
                        
 
    ln.set_data(x[-1], y[-1]) 
    return ln,
# ax = plt.axes()
# ax.set_facecolor('black')
plt.xticks(range(0,10))
plt.yticks(range(0,10))
animation = FuncAnimation(fig, update, interval=100, frames = 16)
animation.save('blue-dot-lec.mp4')
plt.show()
# i = 1
# for x,y in zip(x,y):
#     print(i,x,y)
#     i += 1
#     if i == 17:
#         break
    