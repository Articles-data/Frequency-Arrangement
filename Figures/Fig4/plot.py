import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os        
from matplotlib.ticker import AutoMinorLocator,MaxNLocator, MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

font1 = {'family': 'Times New Roman', 'color': '#000000', 'weight': 'normal'}
fonts = 'Times New Roman'
Number_of_node=1000
Number_of_step=40001#becuase start since 0

def plot(ax, data, step):
    Color_map_hsv = 'hsv'
    Color_map_brg = 'brg'
    Color_of_node = [0 for y in range(Number_of_node + 1)]
    Matrix_sor_0_2PI = [[0 for x in range(Number_of_node)] for y in range(Number_of_step)] 
    for timeforloop in range(0, Number_of_step):
        for y in range(1, Number_of_node + 1):  # for timeforloop step  
            Matrix_sor_0_2PI[timeforloop][y - 1] = data[timeforloop][y] % (2 * math.pi)
    Corolation = [[0 for x in range(Number_of_node)] for y in range(Number_of_node)] 
    Color_of_node[0] = 0
    for x in range(1, Number_of_node):  # Number_of_node+1-2
        Color_of_node[x] = data[step][x] % (2 * math.pi)
    Color_of_node[Number_of_node] = 2 * math.pi
    for x in range(0, Number_of_node):
        for y in range(0, Number_of_node):
            Corolation[x][y] = math.cos(Matrix_sor_0_2PI[step][x] - Matrix_sor_0_2PI[step][y])  # [satr][soton]

    figure = plt.gcf()  # get current figure
    data_Corolation = ax.pcolormesh(Corolation, cmap='jet', vmin=-1, vmax=1)  # binary #hsv
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.set_xlabel('Node', fontdict=font1, fontsize=31)
    ax.set_ylabel('Node', fontdict=font1, fontsize=31, labelpad=-10)
    ax.set_ylim(0,1000)
    ax.set_xlim(0,1000)
    #ax.set_xticks([0.0,0.1, 0.2, 0.3, 0.4])925
    sub_ax = plt.axes([0.925, 0.148, 0.02, 0.704])  # [ , ,arz,ertefa]
    cbar = figure.colorbar(data_Corolation, cax=sub_ax, ticks=np.linspace(-1, 1, 9))
    cbar.set_label('Correlation (D)', fontdict=font1, fontsize=31)
    cbar.ax.tick_params(labelsize=31)

    # Change the y-axis tick labels font
    for label in cbar.ax.get_yticklabels():
        label.set_fontname(fonts)
        label.set_fontsize(31)
        label.set_color('#000000')  # Change the color of the labels


def text_plot(axes):
    annotations = ['(a)', '(b)', '(c)', '(d)','(e)', '(f)', '(g)', '(h)']
    for i, ax in enumerate(axes):
        ax.text(0.94, 0.94, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))

def plot_framework(number_of_pic):
    ax = plt.subplot(2,4,number_of_pic)
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=10)
    plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=5)
    plt.xticks(font=fonts,fontsize=31,color= '#000000')
    plt.yticks(font=fonts,fontsize=31,color= '#000000')
    return ax

fig = plt.figure()
axes=[plot_framework(i) for i in range(1, 9)]



plot(axes[0],np.loadtxt('./Datas/Degree_Radian=1.57_copling=0.09layer1(time)VS(Node).txt'),40)
plot(axes[1],np.loadtxt('./Datas/Degree_Radian=1.57_copling=0.38layer1(time)VS(Node).txt'),22000)
plot(axes[2],np.loadtxt('./Datas/Degree_Radian=1.57_copling=0.39layer1(time)VS(Node).txt'),40000)#good
plot(axes[3],np.loadtxt('./Datas/Degree_Radian=1.567_copling=0.6layer1(time)VS(Node).txt'),4000)#good
plot(axes[4],np.loadtxt('./Datas/Degree_Radian=1.57_copling=1.61layer1(time)VS(Node).txt'),400)#good
plot(axes[5],np.loadtxt('./Datas/Degree_Radian=1.57_copling=2.13layer1(time)VS(Node).txt'),164)#good#good
plot(axes[6],np.loadtxt('./Datas/Degree_Radian=1.57_copling=2.14layer1(time)VS(Node).txt'),610)#good
plot(axes[7],np.loadtxt('./Datas/Degree_Radian=1.57_copling=2.71layer1(time)VS(Node).txt'),40)


text_plot(axes)



y_lim=0.148
x_lim=0.1
plt.subplots_adjust(top = 1-y_lim, bottom=y_lim,left=x_lim-0.05,right=1-x_lim, hspace=0.3, wspace=0.3)


plt.gcf().set_size_inches(28, 16)

plt.savefig("Figure4_dpi100.jpg", dpi=100)
plt.savefig("Figure4_dpi300.png", dpi=300)
plt.savefig("Figure4_dpi300.jpg", dpi=300)
plt.savefig("Figure4.pdf")
with Image.open('Figure4_dpi300.png') as img:
    img.save('Figure4_dpi300.tiff', format='TIFF', compression='tiff_lzw')

#plt.show()