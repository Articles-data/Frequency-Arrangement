import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def Read_Data(address):return np.loadtxt(address)
font1 = {'family': 'Times New Roman', 'color': '#000000', 'weight': 'normal'}
fonts = 'Times New Roman'

def plot_framework(number_of_pic):
    ax = plt.subplot(2,2,number_of_pic)
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
    plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=15)
    plt.xticks(font=fonts,fontsize=31,color= '#000000')
    plt.yticks(font=fonts,fontsize=31,color= '#000000')
    return ax



def plot_ran_T (ax,data_For,data_Bac):
    ax.plot(data_For[:,0], data_For[:,1] ,'o-',color="#3333ff",linewidth=2, markersize=14, markerfacecolor='none')
    ax.plot(data_Bac[:,0], data_Bac[:,1] , linestyle='dashed', marker='o', color="#3333ff",linewidth=2, markersize=8)#, markersize=14, markerfacecolor='none'
    ax.set_ylim(0,1.02)
    ax.set_xlim(0,0.4)
    ax.set_xticks([0.0,0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel('Intralayer coupling strength ($\sigma$)',  fontsize=31, fontdict=font1)
    ax.set_ylabel('Synchronization ($\mathrm{R}^\mathrm{II}$)',  fontsize=31, fontdict=font1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='both', direction='in', which='major', length=22)
    ax.tick_params(axis='both', direction='in', which='minor', length=6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pass




def plot_ran_M (ax,data_For,data_Bac):
    ax.plot(data_For[:,0], data_For[:,3] ,'o-',color="#303030",linewidth=2, markersize=14, markerfacecolor='none')
    ax.plot(data_Bac[:,0], data_Bac[:,3] , linestyle='dashed', marker='o', color="#303030",linewidth=2, markersize=8)#, markersize=14, markerfacecolor='none'
    ax.set_ylim(0,1.02)
    ax.set_xlim(0,0.4)
    ax.set_xticks([0.0,0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel('Intralayer coupling strength ($\sigma$)',  fontsize=31, fontdict=font1)
    ax.set_ylabel('Synchronization ($\mathrm{R}^\mathrm{II}_\mathrm{M}$)',  fontsize=31, fontdict=font1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='both', direction='in', which='major', length=22)
    ax.tick_params(axis='both', direction='in', which='minor', length=6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pass

def plot_ran_L (ax,data_For,data_Bac):
    ax.plot(data_For[:,0], data_For[:,2] ,'o-', color="#00AF00",linewidth=2, markersize=14, markerfacecolor='none')
    ax.plot(data_Bac[:,0], data_Bac[:,2] , linestyle='dashed', marker='o', color="#00AF00",linewidth=2, markersize=8)#, markersize=14, markerfacecolor='none'
    ax.set_ylim(0,1.02)
    ax.set_xlim(0,0.4)
    ax.set_xticks([0.0,0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel('Intralayer coupling strength ($\sigma$)',  fontsize=31, fontdict=font1)
    ax.set_ylabel('Synchronization ($\mathrm{R}^\mathrm{II}_\mathrm{L}$)',  fontsize=31, fontdict=font1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='both', direction='in', which='major', length=22)
    ax.tick_params(axis='both', direction='in', which='minor', length=6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pass

def plot_ran_R (ax,data_For,data_Bac):
    ax.plot(data_For[:,0], data_For[:,4] ,'o-', color="#840000",linewidth=2 , markersize=14, markerfacecolor='none')
    ax.plot(data_Bac[:,0], data_Bac[:,4] , linestyle='dashed', marker='o', color="#840000",linewidth=2, markersize=8)#, markersize=14, markerfacecolor='none'
    ax.set_ylim(0,1.02)
    ax.set_xlim(0,0.4)
    ax.set_xticks([0.0,0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel('Intralayer coupling strength ($\sigma$)',  fontsize=31, fontdict=font1)
    ax.set_ylabel('Synchronization ($\mathrm{R}^\mathrm{II}_\mathrm{R}$)',  fontsize=31, fontdict=font1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='both', direction='in', which='major', length=22)
    ax.tick_params(axis='both', direction='in', which='minor', length=6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pass


def plot_insert(ax,data,final,start,final_y):
    final100=int(final*100)
    start100=int(start*100)
    X = data[start100:final100, 0]
    Y = data[start100:final100, 1]
    Y2 = data[start100:final100, 2]
    ax.plot(X, Y,'o-', color="#3333ff",linewidth=1, markersize=8, markerfacecolor='none')
    ax.plot(X, Y2, linestyle='dashed', marker='o', color="#3333ff",linewidth=1, markersize=4)
    ax.set_ylim(0,final_y)
    ax.set_xlim(start,final)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='both', direction='in', which='major', length=12)
    ax.tick_params(axis='both', direction='in', which='minor', length=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pass


def text_plot(axes):
    annotations = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axes):
        ax.text(0.94, 0.94, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))





def main():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    For=Read_Data('./datas/F.txt')
    Bac=Read_Data('./datas/B.txt')


    fig = plt.figure()
    axes=[plot_framework(i) for i in range(1, 5)]
    plot_ran_T(axes[0],For,Bac)
    plot_ran_L(axes[1],For,Bac)
    plot_ran_M(axes[2],For,Bac)
    plot_ran_R(axes[3],For,Bac)


    text_plot(axes)

    x_lim=0.246
    plt.subplots_adjust(top = 0.97, bottom=0.08,left=x_lim,right=1-x_lim, hspace=0.3, wspace=0.3)
    plt.gcf().set_size_inches(28, 16)


    plt.savefig("FigureS1_dpi100.jpg", dpi=100)
    plt.savefig("FigureS1_dpi300.png", dpi=300)
    plt.savefig("FigureS1_dpi300.jpg", dpi=300)
    from PIL import Image
    with Image.open('FigureS1_dpi300.png') as img:
        img.save('FigureS1-dpi300.tiff', format='TIFF', compression='tiff_lzw')
    plt.savefig("FigureS1.pdf")
    pass

if __name__=="__main__":
    main()