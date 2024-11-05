import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from PIL import Image

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


def plot_ran (ax,data):
    ax.plot(data[:,0], data[:,1] ,'o-', color="#3333ff",linewidth=2, markersize=14, markerfacecolor='none')
    ax.plot(data[:,0], data[:,2] , linestyle='dashed', marker='o', color="#3333ff",linewidth=2, markersize=8)
    ax.set_ylim(0,1.02)
    ax.set_xlim(0,3)
    ax.set_xlabel('Intralayer coupling strength ($\sigma$)',  fontsize=31, fontdict=font1)
    ax.set_ylabel('Synchronization ($\mathrm{R}^\mathrm{II}$)',  fontsize=31, fontdict=font1)
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
    # insert
    ax.plot(X, Y,'o-', color="#3333ff",linewidth=1, markersize=8, markerfacecolor='none')
    ax.plot(X, Y2, linestyle='dashed', marker='o', color="#3333ff",linewidth=1, markersize=4)
    ax.set_ylim(0,final_y)
    ax.set_xlim(start,final)
    final0=final/3
    ax.set_xticks([0.0,final0, final0*2, final0*3])
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='both', direction='in', which='major', length=12)
    ax.tick_params(axis='both', direction='in', which='minor', length=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    pass

def plot_framework2():
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
    plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=15)
    plt.xticks(font=fonts,fontsize=20,color= '#000000')
    plt.yticks(font=fonts,fontsize=20,color= '#000000')
    pass


def text_plot(axes):
    # Add text annotations for each subplot
    annotations = ['(a)', '(b)', '(c)', '(d)']
    
    for i, ax in enumerate(axes):
        ax.text(0.94, 0.94, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))


def text_plot_box(axes):
    # Add text annotations for each subplot
    annotations = ['Random Model\n          α=0', 'Random Model\n        α=π/2', 'Regular Model\n         α=0', 'Regular Model\n       α=π/2']
    
    for i, ax in enumerate(axes):
        if i==0:
            ax.text(0.35, 0.84, annotations[0], fontdict=font1, fontsize=30,
                    transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))
        if i==1:
            ax.text(0.08, 0.84, annotations[1], fontdict=font1, fontsize=30,
                    transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))
        if i==2:
            ax.text(0.35, 0.84, annotations[2], fontdict=font1, fontsize=30,
                    transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))
        if i==3:
            ax.text(0.08, 0.84, annotations[3], fontdict=font1, fontsize=30,
                    transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))
    






def main():
    # Set the font properties
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    L_U=Read_Data('./a/r0_0to3.txt')
    R_U=Read_Data('./b/r70_0to3.txt')
    L_D=Read_Data('./c/s0_0to3.txt')
    R_D=Read_Data('./d/s70fb.txt')
    fig = plt.figure()

    axes=[plot_framework(i) for i in range(1, 5)]
    plot_ran(axes[0],L_U)
    plot_ran(axes[1],R_U)
    plot_ran(axes[2],L_D)
    plot_ran(axes[3],R_D)



    axes3 = fig.add_axes([0.31, 0.634, 0.12, 0.2]) # inset axes
    plot_framework2()
    plot_insert(axes3,L_U,0.6,0,1)


    axes2 = fig.add_axes([0.31, 0.13, 0.12, 0.2]) # inset axes
    plot_framework2()
    plot_insert(axes2,L_D,0.3,0,1)



    text_plot(axes)
    text_plot_box(axes)


    x_lim=0.246
    plt.subplots_adjust(top = 0.97, bottom=0.08,left=x_lim,right=1-x_lim, hspace=0.3, wspace=0.3)


    plt.gcf().set_size_inches(28, 16)

    plt.savefig("Figure3_dpi100.jpg", dpi=100)
    plt.savefig("Figure3_dpi300.png", dpi=300)
    plt.savefig("Figure3_dpi300.jpg", dpi=300)
    plt.savefig("Figure3.pdf")
    with Image.open('Figure3_dpi300.png') as img:
        img.save('Figure3_dpi300.tiff', format='TIFF', compression='tiff_lzw')
    pass

if __name__=="__main__":
    main()