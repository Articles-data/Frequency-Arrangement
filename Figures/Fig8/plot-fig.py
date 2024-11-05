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
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def plot_framework(number_of_pic):
        ax = plt.subplot(3,2,number_of_pic)
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=15)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
        return ax

def text_plot(axes):
        annotations = ['(a)', '(b)', '(c)', '(d)','(e)', '(f)']
        for i, ax in enumerate(axes):
                ax.text(0.06, 0.9, annotations[i], fontdict=font1, fontsize=30,
                        transform=ax.transAxes, ha='left', va='top',
                        bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))

def plot(ax,datax,datay,lim1,lim2,col):  
        ax.plot(datax, datay, color=col,linewidth=3)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
        ax.set_xlabel('Intralayer coupling strength ($\sigma$)', fontdict=font1, fontsize=31)
        ax.set_ylabel('Synchronization ($\mathrm{r}^\mathrm{II}$)', fontdict=font1, fontsize=31)
        ax.set_ylim([0, 1])
        ax.set_xlim([lim1, lim2])
        ax.tick_params(axis='both', direction='in', which='major', length=22)
        ax.tick_params(axis='both', direction='in', which='minor', length=6)

def plot_d(ax,data2,lim1,lim2,gstring):  
        #ax.plot(datax, datay, color=col,linewidth=3)
        ax.plot(data2[:,0],data2[:,1],'o-', color='#00CC00',linewidth=3, markersize=10)
        ax.tick_params(axis='y', labelcolor='#004300')
        ax2 = ax.twinx()
        ax2.plot(data2[:,0],data2[:,2],'^-', color='#840000',linewidth=3, markersize=12)
        ax2.tick_params(axis='y', labelcolor='#430000')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
        ax2.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)

        ax.set_xlabel(gstring, fontdict=font1, fontsize=31)
        ax.set_ylabel('First transition point ($\sigma^\mathrm{1}_\mathrm{T}$)', fontdict=font1, fontsize=31, color='#004300')
        ax2.set_ylabel('Second transition point ($\sigma^\mathrm{2}_\mathrm{T}$)', fontdict=font1, fontsize=31, color='#430000')
        #ax.set_ylim([0, 1])
        #ax.set_xlim([lim1, lim2])
        #ax.set_ylim(0,1.02)
        if gstring=="Interlayer coupling strength ($\lambda$)":
                ax.set_xlim(1,23)
                ax.set_xticks([2,4, 6,8,10,12,14,16,18,20,22])
                #ax.set_ylim(0.2,0.6)
                ax.set_ylim(0.2,0.66)
                ax2.set_ylim(1,3)

        if gstring=="Average interlayer frequency difference ($\Delta\omega$)":
                ax.set_xlim(0.05,0.95)
                ax.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
                #ax.set_ylim(0.28,0.44)
                #ax.set_yticks([0.28,0.36,0.44])
                ax.set_ylim(0.2,0.66)
                ax2.set_ylim(1,3)

        if gstring=="Frustration (\u03B1)":
                ax.set_xlim(1.5385,1.5715)
                #ax.set_xticks([1.54,1.543,1.546,1.549,1.552,1.555,1.558,1.561,1.564,1.567,1.57])
                #ax.set_ylim(0.35,0.65)
                #ax.set_yticks([0.35,0.5,0.65])
                ax.set_ylim(0.2,0.66)
                ax2.set_ylim(1,3)


        ax.tick_params(axis='both', direction='in', which='major', length=22)
        ax.tick_params(axis='both', direction='in', which='minor', length=6)
        ax.tick_params(axis='y', labelcolor='#000000', labelsize=31)
        ax2.tick_params(axis='y',labelcolor='#000000', labelsize=31)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))




def main():
        fig = plt.figure()
        axes=[plot_framework(i) for i in range(1, 7)]




        data=np.loadtxt('./l.txt')
        plot(axes[0],data[:,0],data[:,1],0,3,"#38ACEC")
        plot(axes[0],data[:,0],data[:,2],0,3,"#157DEC")
        plot(axes[0],data[:,0],data[:,3],0,3,"#1974D2")
        plot(axes[0],data[:,0],data[:,4],0,3,"#1569C7")
        plot(axes[0],data[:,0],data[:,5],0,3,"#2554C7")#2554C7	Sapphire Blue
        plot(axes[0],data[:,0],data[:,6],0,3,"#0041C2")#0041C2	Blueberry Blue
        plot(axes[0],data[:,0],data[:,7],0,3,"#0020C2")#0020C2	Cobalt Blue
        plot(axes[0],data[:,0],data[:,8],0,3,"#0000A5")#0000A5	Earth Blue
        plot(axes[0],data[:,0],data[:,9],0,3,"#000080")#00008B	DarkBlue (W3C)
        plot(axes[0],data[:,0],data[:,10],0,3,"#191970")#191970	MidnightBlue (W3C)
        plot(axes[0],data[:,0],data[:,11],0,3,"#151B54")#151B54	Night Blue

        data=np.loadtxt('./w.txt')
        plot(axes[2],data[:,0],data[:,1],0,3,"#38ACEC")
        plot(axes[2],data[:,0],data[:,2],0,3,"#157DEC")
        plot(axes[2],data[:,0],data[:,3],0,3,"#1974D2")
        plot(axes[2],data[:,0],data[:,4],0,3,"#1569C7")
        plot(axes[2],data[:,0],data[:,5],0,3,"#2554C7")#2554C7	Sapphire Blue
        plot(axes[2],data[:,0],data[:,6],0,3,"#0041C2")#0041C2	Blueberry Blue
        plot(axes[2],data[:,0],data[:,7],0,3,"#0020C2")#0020C2	Cobalt Blue
        plot(axes[2],data[:,0],data[:,8],0,3,"#0000A5")#0000A5	Earth Blue
        plot(axes[2],data[:,0],data[:,9],0,3,"#000080")#00008B	DarkBlue (W3C)
        #plot(ax_3,data[:,0],data[:,10],0,3,"#191970")#191970	MidnightBlue (W3C)
        #plot(ax_3,data[:,0],data[:,11],0,3,"#151B54")#151B54	Night Blue

        data=np.loadtxt('./a.txt')
        plot(axes[4],data[:,0],data[:,1],0,3,"#38ACEC")
        plot(axes[4],data[:,0],data[:,2],0,3,"#157DEC")
        plot(axes[4],data[:,0],data[:,3],0,3,"#1974D2")
        plot(axes[4],data[:,0],data[:,4],0,3,"#1569C7")
        plot(axes[4],data[:,0],data[:,5],0,3,"#2554C7")#2554C7	Sapphire Blue
        plot(axes[4],data[:,0],data[:,6],0,3,"#0041C2")#0041C2	Blueberry Blue
        plot(axes[4],data[:,0],data[:,7],0,3,"#0020C2")#0020C2	Cobalt Blue
        plot(axes[4],data[:,0],data[:,8],0,3,"#0000A5")#0000A5	Earth Blue
        plot(axes[4],data[:,0],data[:,9],0,3,"#000080")#00008B	DarkBlue (W3C)
        plot(axes[4],data[:,0],data[:,10],0,3,"#191970")#191970	MidnightBlue (W3C)
        plot(axes[4],data[:,0],data[:,11],0,3,"#151B54")#151B54	Night Blue





        plot_d(axes[1],np.loadtxt('./ld.txt'),0,3,"Interlayer coupling strength ($\lambda$)")
        plt.yticks(font=fonts,fontsize=31,color= '#000000')

        plot_d(axes[3],np.loadtxt('./w2.txt'),0,3,"Average interlayer frequency difference ($\Delta\omega$)")
        plt.yticks(font=fonts,fontsize=31,color= '#000000')

        plot_d(axes[5],np.loadtxt('./a2.txt'),0,3,"Frustration (\u03B1)")
        plt.yticks(font=fonts,fontsize=31,color= '#000000')


        text_plot(axes)

        y_lim=0.06
        x_lim=0.08
        plt.subplots_adjust(top = 1-y_lim, bottom=y_lim+0.02,left=x_lim,right=1-x_lim, hspace=0.4, wspace=0.3)
        plt.gcf().set_size_inches(28, 20)

        '''plt.savefig('./Figure9_100.pdf.png', dpi=100, bbox_inches='tight', pad_inches=1, bbox_extra_artists=[])
        plt.savefig('./Figure9_500.pdf.png', dpi=500, bbox_inches='tight', pad_inches=1, bbox_extra_artists=[])
        plt.savefig('./Figure9.pdf', bbox_inches='tight', pad_inches=1, bbox_extra_artists=[])'''

        #plt.savefig("Figure8_dpi100_3.jpg", dpi=100, bbox_inches='tight', pad_inches=1, bbox_extra_artists=[])  
        plt.savefig("Figure8_dpi100.jpg", dpi=100)
        plt.savefig("Figure8_dpi300.png", dpi=300)
        plt.savefig("Figure8_dpi300.jpg", dpi=300)
        plt.savefig("Figure8.pdf")
        with Image.open('Figure8_dpi300.png') as img:
                img.save('Figure8_dpi300.tiff', format='TIFF', compression='tiff_lzw')
pass

if __name__=="__main__":
        main()