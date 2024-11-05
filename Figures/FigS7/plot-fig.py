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

def text_plot(axes):
        annotations = ['(a)', '(b)', '(c)', '(d)']
        for i, ax in enumerate(axes):
                ax.text(0.06, 0.9, annotations[i], fontdict=font1, fontsize=30,
                        transform=ax.transAxes, ha='left', va='top',
                        bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))


def plot(ax,Matrix,time,ioz,c1,c2):
        a = np.zeros(Number_of_step)
        b = np.zeros(Number_of_step)
        
        for t in range(0, Number_of_step):
                a[t]=Matrix[t][ioz]
                b[t]=Matrix[t][999-ioz]
        ax.set_xlim([100, 200])
        ax.set_ylim([-1.04, 1.04])
        ax.set_yticks([-1.0,-0.5,0,0.5,1])
        colors = []
        line, = ax.plot(time, a, color=c1, label=f"${ioz+1}^{{\\mathrm{{th}}}}$ node", linewidth=2.6)
        colors.append(plt.getp(line,'color'))
        line, = ax.plot(time, b, color=c2, label=f"${999-ioz+1}^{{\\mathrm{{th}}}}$ node", linewidth=2.6)
        colors.append(plt.getp(line,'color'))
        
        ax.set_xlabel('Time (t)', fontdict=font1, fontsize=30)
        ax.set_ylabel('Mirror nodes correlation', fontdict=font1, fontsize=30)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
        
        ax.tick_params(axis='both', direction='in', which='major', length=22)
        ax.tick_params(axis='both', direction='in', which='minor', length=6)
        legend = ax.legend(loc=(1),fontsize=31)
        
        for color,text in zip(colors,legend.get_texts()):
                text.set_color(color)
                text.set_fontname(fonts)
        # Set the background color and transparency for the legend
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.94)  # Transparency level (0.3 for 30%)

def read_data(Lx):
        Matrix_sor_0_2PI = [[0 for x in range(Number_of_node)] for y in range(Number_of_step)] 
        Degree=1.57 #degree alpha
        Degree_data='Degree_Radian='+str(Degree)#string degree
        copling=2.11 #coupling in layers
        file_name=Degree_data+'_copling='+str(copling)+'layer'+Lx+'(time)VS(Node)'
        address=r'./Data/Forward/'
        data=np.loadtxt(address+'Save/Phases/'+file_name+'.txt')
        for timeforloop in range(0, Number_of_step):
                for y in range(1, Number_of_node+1):#for timeforloop step  
                        Matrix_sor_0_2PI[timeforloop][y-1]=data[timeforloop][y]%(2*math.pi)
        return Matrix_sor_0_2PI

def call_matrix(data2,data1):
        Matrix = [[0 for x in range(Number_of_node)] for y in range(Number_of_step)] 
        time = np.arange(0, 400.01, 0.01)
        for t in range(0, Number_of_step):
                for y in range(0, Number_of_node):
                        Matrix[t][y]=np.cos(data2[t][y] - data1[t][y])
        return Matrix,time


def plot_framework(number_of_pic):
        ax = plt.subplot(2,2,number_of_pic)
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=15)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
        return ax

def main():


        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
        
        fig = plt.figure()
        axes=[plot_framework(i) for i in range(1, 5)]
        
        
        plt.xticks(np.arange(100, 220, 20),font=fonts,fontsize=31,color= '#000000')
        plt.yticks(np.arange(-1, 1.5, 0.5),font=fonts,fontsize=31,color= '#000000')




        Data2=read_data('2')
        Data1=read_data('1')


        matrix,time=call_matrix(Data2,Data1)

        plot(axes[0],matrix,time,0,"#00CC00","#840000")
        plot(axes[1],matrix,time,100,"#00CC00","#840000")
        plot(axes[2],matrix,time,200,"#00CC00","#840000")
        plot(axes[3],matrix,time,400,"#303030","#303030")



        text_plot(axes) 

        y_lim=0.22
        x_lim=0.08
        plt.subplots_adjust(top = 1-y_lim, bottom=y_lim+0.02,left=x_lim,right=1-x_lim, hspace=0.4, wspace=0.3)
        plt.gcf().set_size_inches(28, 20)
        #plt.subplots_adjust( hspace=0.5, wspace=0.4)


        plt.savefig("FigureS7_dpi100.jpg", dpi=100)
        plt.savefig("FigureS7_dpi300.png", dpi=300, bbox_inches='tight', pad_inches=1, bbox_extra_artists=[])
        plt.savefig("FigureS7_dpi300.jpg", dpi=300, bbox_inches='tight', pad_inches=1, bbox_extra_artists=[])
        plt.savefig("FigureS7.pdf")
        with Image.open('FigureS7_dpi300.png') as img:
                img.save('FigureS7_dpi300.tiff', format='TIFF', compression='tiff_lzw')
pass

if __name__=="__main__":
        main()