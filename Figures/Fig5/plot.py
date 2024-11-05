import os #for create empty folder
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import matplotlib as mpl
import math
import cmath
import time
from PIL import Image

import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.colors
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection
import gc
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoMinorLocator,MaxNLocator, MultipleLocator

Number_of_node=1000
Number_of_step=40001#becuase start since 0

Color_map_hsv='hsv'
Color_map_brg='brg'
address=r'./Data/Forward/'
landa=10
dW=0.8
dw_left_cut=277
dw_Right_cut=723

Matrix_sor_0_2PI = [[0 for x in range(Number_of_node)] for y in range(Number_of_step)] 
Corolation = [[0 for x in range(Number_of_node)] for y in range(Number_of_node)] 
#Part syncroney
Number_total=[0 for y in range(Number_of_step)]#total sync
Total_sync=[0 for y in range(Number_of_step)]#total sync
left_sync=[0 for y in range(Number_of_step)]
mid_sync=[0 for y in range(Number_of_step)]
right_sync=[0 for y in range(Number_of_step)]




Degree=1.57 #degree alpha
Degree_data='Degree_Radian='+str(Degree)#string degree



copling=0.39 #coupling in layers

#for coupling_forloop in np.arange(0.44, 0.64, 0.01):
    #copling = np.round(coupling_forloop, 2)  # Round to 2 decimal places
file_name=Degree_data+'_copling='+str(copling)+'layer2(time)VS(Node)'
################################################################################
#                           Read data Phase nodes                      # START #
################################################################################
data=np.loadtxt(address+'Save/Phases/'+file_name+'.txt')
#read calculation Matrix_sor_0_2PI
for timeforloop in range(0, Number_of_step):
    for y in range(1, Number_of_node+1):#for timeforloop step  
        Matrix_sor_0_2PI[timeforloop][y-1]=data[timeforloop][y]%(2*math.pi)
################################################################################
#                           Read data Phase nodes                      #  END  #
################################################################################
################################################################################
#                                   Part syncroney                     # START #
################################################################################
for timeforloop in range(0, Number_of_step):
    rc = 0.0
    rs = 0.0
    for y in range(0, Number_of_node):#for timeforloop step  
        rc=rc+math.cos(Matrix_sor_0_2PI[timeforloop][y])
        rs=rs+math.sin(Matrix_sor_0_2PI[timeforloop][y])
    Number_total[timeforloop]=(timeforloop/100)#doroste
    Total_sync[timeforloop]=(math.sqrt(math.pow(rc, 2) + math.pow(rs, 2)) / (Number_of_node))

for timeforloop in range(0, Number_of_step):
    rc = 0.0
    rs = 0.0
    for y in range(0,dw_left_cut):#for timeforloop step  
        rc=rc+math.cos(Matrix_sor_0_2PI[timeforloop][y])
        rs=rs+math.sin(Matrix_sor_0_2PI[timeforloop][y])
    left_sync[timeforloop]=(math.sqrt(math.pow(rc, 2) + math.pow(rs, 2)) / (dw_left_cut))    


for timeforloop in range(0, Number_of_step):
    rc = 0.0
    rs = 0.0
    for y in range(dw_left_cut,dw_Right_cut):#for timeforloop step  
        rc=rc+math.cos(Matrix_sor_0_2PI[timeforloop][y])
        rs=rs+math.sin(Matrix_sor_0_2PI[timeforloop][y])
    mid_sync[timeforloop]=(math.sqrt(math.pow(rc, 2) + math.pow(rs, 2)) / (dw_Right_cut-dw_left_cut))    


for timeforloop in range(0, Number_of_step):
    rc = 0.0
    rs = 0.0
    for y in range(dw_Right_cut,Number_of_node):#for timeforloop step  
        rc=rc+math.cos(Matrix_sor_0_2PI[timeforloop][y])
        rs=rs+math.sin(Matrix_sor_0_2PI[timeforloop][y])
    right_sync[timeforloop]=(math.sqrt(math.pow(rc, 2) + math.pow(rs, 2)) / (Number_of_node-dw_Right_cut))   
#for step in range(0, 20000,100):#Number_of_node+1-2
font1 = {'family': 'Times New Roman', 'color': '#000000', 'weight': 'normal'}
fonts = 'Times New Roman'

start_time=time.time()

# Set the font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'






def plot_framework(number_of_pic):
    if number_of_pic==1:
        ax = plt.subplot(2,2,(1,2))
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=8)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
    else:
        ax = plt.subplot(2,2,number_of_pic+1)
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=8)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
    return ax


def text_plot(axes):
    # Add text annotations for each subplot
    annotations = ['(a)', '(b)', '(c)']
    
    for i, ax in enumerate(axes):
        if i==0:
            ax.text(0.028, 0.94, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))
        else:
            ax.text(0.06, 0.94, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))



def plot(ax, data, step):

    Matrix_sor_0_2PI = [[0 for x in range(Number_of_node)] for y in range(Number_of_step)] 
    for timeforloop in range(0, Number_of_step):
        for y in range(1, Number_of_node + 1):  # for timeforloop step  
            Matrix_sor_0_2PI[timeforloop][y - 1] = data[timeforloop][y] % (2 * math.pi)
    Corolation = [[0 for x in range(Number_of_node)] for y in range(Number_of_node)] 

    for x in range(0, Number_of_node):
        for y in range(0, Number_of_node):
            Corolation[x][y] = math.cos(Matrix_sor_0_2PI[step][x] - Matrix_sor_0_2PI[step][y])  # [satr][soton]

    figure = plt.gcf()  # get current figure
    data_Corolation = ax.pcolormesh(Corolation, cmap='jet', vmin=-1, vmax=1)  # binary #hsv
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.set_xlabel('Node', fontdict=font1, fontsize=31)
    ax.set_ylabel('Node', fontdict=font1, fontsize=31, labelpad=-6)
    if step==32100:
        sub_ax = plt.axes([0.774, 0.08, 0.020, 0.387])  # [ , ,arz,ertefa]
        cbar = figure.colorbar(data_Corolation, cax=sub_ax, ticks=np.linspace(-1, 1, 9))
        cbar.set_label('Correlation (D)', fontdict=font1, fontsize=31)
        cbar.ax.tick_params(labelsize=31)

        # Change the y-axis tick labels font
        for label in cbar.ax.get_yticklabels():
            label.set_fontname(fonts)
            label.set_fontsize(31)
            label.set_color('#000000')  # Change the color of the labels


fig = plt.figure()

axes=[plot_framework(i) for i in range(1, 4)]


'''fig = plt.figure()
axes[0] = plt.subplot(2, 2, (1,2))
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=15)
plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, pad=15)
plt.xticks(font=fonts,fontsize=31,color= '#000000')
plt.yticks(font=fonts,fontsize=31,color= '#000000')
axes[1] = plt.subplot(2, 2, 3)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=15)
plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, pad=15)
plt.xticks(font=fonts,fontsize=31,color= '#000000')
plt.yticks(font=fonts,fontsize=31,color= '#000000')
axes[2] = plt.subplot(2, 2, 4)
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=15)
plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, pad=15)
plt.xticks(font=fonts,fontsize=31,color= '#000000')
plt.yticks(font=fonts,fontsize=31,color= '#000000')'''




step1=5600
step2=32100




colors = []

line, = axes[0].plot(Number_total, savgol_filter(Total_sync, 70, 2),  color='#0000CD', label='$\mathrm{r}^\mathrm{II}$', linewidth = '2.8')#3333ff
colors.append(plt.getp(line,'color'))
line, = axes[0].plot(Number_total, savgol_filter(left_sync, 70, 2), color="#00CC00", label='$\mathrm{r}^\mathrm{II}_\mathrm{L}$', linewidth = '2.8')#620000#FF7F0E
colors.append(plt.getp(line,'color'))
line, = axes[0].plot(Number_total, savgol_filter(mid_sync, 70, 2), color="#303030", label='$\mathrm{r}^\mathrm{II}_\mathrm{M}$', linewidth = '2.8')
colors.append(plt.getp(line,'color'))
line, = axes[0].plot(Number_total, savgol_filter(right_sync, 70, 2), color="#840000", label='$\mathrm{r}^\mathrm{II}_\mathrm{R}$', linewidth = '2.8')
colors.append(plt.getp(line,'color'))





axes[0].axvline(x=step1/100,color = '#000000', linestyle='dashed',linewidth='2.8')
axes[0].axvline(x=step2/100,color = '#000000', linestyle='dashed',linewidth='2.8')




axes[0].set_xlabel('Time (t)', fontdict=font1, fontsize=30, labelpad=20)


axes[0].set_ylabel('Synchronization ($\mathrm{r}^\mathrm{II}$)', fontdict=font1, fontsize=30, labelpad=29)
axes[0].set_xlim([0, 400])
axes[0].set_ylim([0, 1])
axes[0].xaxis.set_minor_locator(AutoMinorLocator())
axes[0].yaxis.set_minor_locator(AutoMinorLocator())

axes[0].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)

axes[0].tick_params(axis='both', direction='in', which='major', length=22)
axes[0].tick_params(axis='both', direction='in', which='minor', length=6)
L=axes[0].legend(loc=(1.04,0.28),fontsize=32)
for color,text in zip(colors,L.get_texts()):
    text.set_color(color)
    text.set_fontname(fonts)

plt.setp(L.texts, family='Times New Roman')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=15)
plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, pad=15)
plt.xticks(font=fonts,fontsize=31,color= '#000000')
plt.yticks(font=fonts,fontsize=31,color= '#000000')

axes[0].text(51.299, 1.02, ' b', fontsize=31, color='#000000', fontname='Times New Roman')

axes[0].text(316.439, 1.02, ' c', fontsize=31, color='#000000', fontname='Times New Roman')
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#axes[1] = fig.add_subplot(2, 2, 3)
'''for x in range(0, Number_of_node):
        for y in range(0, Number_of_node):
            Corolation[x][y]=math.cos(Matrix_sor_0_2PI[step1][x]-Matrix_sor_0_2PI[step1][y])#[satr][soton]

#figure = plt.gcf()  # get current figure
data_Corolation = axes[1].pcolormesh(Corolation,cmap='jet', vmin=-1, vmax=1)#binary #hsv
sub_ax= plt.axes([0.732, 0.08, 0.02, 0.393]) # [ , ,arz,ertefa]
cbar =fig.colorbar(data_Corolation, cax=sub_ax, ticks=np.linspace(-1,1, 9))
cbar.set_label('Correlation', fontdict=font1, fontsize=31, labelpad=17)
axes[1].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
axes[1].set_xlabel('Node', fontdict=font1, fontsize=31, labelpad=20)
axes[1].set_ylabel('Node', fontdict=font1, fontsize=31, labelpad=10)
#axes[1].text(480, 1044, r'$\mathbf{A}$', fontsize=51, color='#000000', fontname='Arial')

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#axes[2] = fig.add_subplot(2, 2, 4)

for x in range(0, Number_of_node):
        for y in range(0, Number_of_node):
            Corolation[x][y]=math.cos(Matrix_sor_0_2PI[step2][x]-Matrix_sor_0_2PI[step2][y])#[satr][soton]

#figure = plt.gcf()  # get current figure
figure = plt.gcf()  # get current figure
data_Corolation = axes[2].pcolormesh(Corolation, cmap='jet', vmin=-1, vmax=1)  # binary #hsv
#sub_axes[2] = plt.axes([0.732, 0.08, 0.02, 0.393]) # [ , ,arz,ertefa]
cbar = figure.colorbar(data_Corolation, cax=sub_ax[1], ticks=np.linspace(-1, 1, 9))
cbar.set_label('Correlation (D)', fontdict=font1, fontsize=31, labelpad=17)
cbar.ax.tick_params(labelsize=31)#cbar.set_label('Correlation')
axes[2].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
axes[2].set_xlabel('Node', fontdict=font1, fontsize=31, labelpad=20)
axes[2].set_ylabel('Node', fontdict=font1, fontsize=31, labelpad=10)
#axes[2].text(480, 1044, r'$\mathbf{B}$', fontsize=51, color='#000000', fontname='Arial')

# Change the y-axis tick labels font
for label in cbar.ax.get_yticklabels():
    label.set_fontname(fonts)
    label.set_fontsize(31)
    label.set_color('#000000')  # Change the color of the labels'''


##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


plot(axes[1],np.loadtxt('./Data/Forward/Save/Phases/Degree_Radian=1.57_copling=0.39layer2(time)VS(Node).txt'),step1)#good
plot(axes[2],np.loadtxt('./Data/Forward/Save/Phases/Degree_Radian=1.57_copling=0.39layer2(time)VS(Node).txt'),step2)


text_plot(axes)



x_lim=0.246
plt.subplots_adjust(top = 0.97, bottom=0.08,left=x_lim,right=1-x_lim, hspace=0.3, wspace=0.3)


plt.gcf().set_size_inches(28, 16)





plt.savefig("Figure5_dpi100.jpg", dpi=100)
plt.savefig("Figure5_dpi300.png", dpi=300)
plt.savefig("Figure5_dpi300.jpg", dpi=300)
plt.savefig("Figure5.pdf")
with Image.open('Figure5_dpi300.png') as img:
    img.save('Figure5_dpi300.tiff', format='TIFF', compression='tiff_lzw')

print(str(step1)+"=> saving time: {:.3f} sec".format(time.time()-start_time))
#plt.show()
plt.close(fig)
gc.collect()