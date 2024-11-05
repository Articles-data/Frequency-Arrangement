import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os        
from matplotlib.ticker import AutoMinorLocator,MaxNLocator, MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

# Set the font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
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
    ax.set_ylabel('Node', fontdict=font1, fontsize=31, labelpad=-6)

    sub_ax = plt.axes([0.785, 0.08, 0.028, 0.89])  # [ , ,arz,ertefa]
    cbar = figure.colorbar(data_Corolation, cax=sub_ax, ticks=np.linspace(-1, 1, 9))
    cbar.set_label('Correlation (D)', fontdict=font1, fontsize=31)
    cbar.ax.tick_params(labelsize=31)

    # Change the y-axis tick labels font
    for label in cbar.ax.get_yticklabels():
        label.set_fontname(fonts)
        label.set_fontsize(31)
        label.set_color('#000000')  # Change the color of the labels




def plot_framework(number_of_pic):
    ax = plt.subplot(2,2,number_of_pic)
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
    plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=8)
    plt.xticks(font=fonts,fontsize=31,color= '#000000')
    plt.yticks(font=fonts,fontsize=31,color= '#000000')
    return ax

def text_plot(axes):
    # Add text annotations for each subplot
    annotations = ['(a)', '(b)', '(c)', '(d)']
    
    for i, ax in enumerate(axes):
        ax.text(0.94, 0.94, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))




def main():


    fig = plt.figure()

    axes=[plot_framework(i) for i in range(1, 5)]
    plot(axes[0],np.loadtxt('./Datas/a=0.000_k=0.50layer2.txt'),4000)#good
    plot(axes[1],np.loadtxt('./Datas/a=0.000_k=0.25layer2.txt'),40000)#good
    plot(axes[2],np.loadtxt('./Datas/a=0.000_k=0.15layer2.txt'),22000)
    plot(axes[3],np.loadtxt('./Datas/a=0.000_k=0.00layer2.txt'),40)


    text_plot(axes)



    x_lim=0.246
    plt.subplots_adjust(top = 0.97, bottom=0.08,left=x_lim,right=1-x_lim, hspace=0.3, wspace=0.3)


    plt.gcf().set_size_inches(28, 16)

    plt.savefig("FigureS2_dpi100.jpg", dpi=100)

    plt.savefig("FigureS2_dpi300.png", dpi=300)
    plt.savefig("FigureS2_dpi300.jpg", dpi=300)
    from PIL import Image
    with Image.open('FigureS2_dpi300.png') as img:
        img.save('FigureS2-dpi300.tiff', format='TIFF', compression='tiff_lzw')
    plt.savefig("FigureS2.pdf")
    pass

if __name__=="__main__":
    main()