from matplotlib.ticker import AutoMinorLocator,MaxNLocator, MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def Read_Data(address):return np.loadtxt(address)
font1 = {'family': 'Times New Roman', 'color': '#000000', 'weight': 'normal'}
fonts = 'Times New Roman'


def plot_eshel(number_of_pic):
    ax = plt.subplot(2,2,number_of_pic)
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
    plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=15)
    plt.xticks(font=fonts,fontsize=31,color= '#000000')
    plt.yticks(font=fonts,fontsize=31,color= '#000000')
    return ax

def plot_ran(ax,x,y_l1,y_l2):  
    #font1 = {'family': 'Arial', 'color': '#262626'}
    ax.scatter(x, y_l1,  color="#3333ff", s=160,alpha=1, edgecolor='k')
    ax.plot(x, y_l2  , color="k",linewidth=7.8)
    ax.plot(x, y_l2  , color="#ff3333",linewidth=6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #ax.xaxis.set_major_locator(MaxNLocator())
    #ax.yaxis.set_major_locator(MaxNLocator())#nbins=2
    ax.set_xlabel('Node', fontsize=31, fontdict=font1)
    ax.set_ylabel('Natural frequency ($\omega$)', fontsize=31, fontdict=font1)
    ax.set_xlim([0, 1000])
    ax.set_yticks(ticks=np.arange(-0.75, 0.75, 0.25))
    ax.set_ylim([-0.5, 0.5])
    ax.tick_params(axis='both', direction='in', which='major', length=22)
    ax.tick_params(axis='both', direction='in', which='minor', length=6)
def plot_reg(ax,x,y_l1,y_l2):  
    #font1 = {'family': 'Arial', 'color': '#262626'}
    ax.scatter(x, y_l1,  color="k", s=229,alpha=1)
    ax.scatter(x, y_l1,  color="#3333ff", s=160,alpha=1)
    ax.plot(x, y_l2  , color="k",linewidth=7.8)
    ax.plot(x, y_l2  , color="#ff3333",linewidth=6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('Node',  fontsize=31, fontdict=font1)
    ax.set_ylabel('Natural frequency ($\omega$)', fontsize=31, fontdict=font1)
    ax.set_xlim([0, 1000])
    ax.set_yticks(ticks=np.arange(-0.75, 0.75, 0.25))
    ax.set_ylim([-0.5, 0.5])

    ax.tick_params(axis='both', direction='in', which='major', length=22)
    ax.tick_params(axis='both', direction='in', which='minor', length=6)



def plot_ran_hist(ax,y_l1,y_l2):  
    def get_histogram_values(data, num_bins):
        fig1,ax1 = plt.subplots()
        n, bins, _ = ax1.hist(data, bins=num_bins)
        plt.close(fig1)
        return n, bins
    #font1 = {'family': 'Arial', 'color': '#262626'}
    dw_rand=[0 for y in range(1000)]
    for i in range(0,1000):
        dw_rand[i]=2*abs(y_l1[i]-y_l2[i])
    bins=[]
    n=[]
    n, bins = get_histogram_values(dw_rand, 20)
    bins = [((i*0.1)+0.05) for i in range(20)]
    n=n/1000
    ax.bar(bins, n, width=0.098, color="#3333ff", edgecolor='k', alpha=1, linewidth=1.3)
    ax.set_ylim(0, .5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('Interlayer frequency differences (2$×|\delta\omega|$)',  fontsize=31, fontdict=font1)
    ax.set_ylabel('Probability density function',  fontsize=31, fontdict=font1)
    #ax.set_xlim([0, 2])
    ax.tick_params(axis='both', direction='in', which='major', length=22)
    ax.tick_params(axis='both', direction='in', which='minor', length=6)
    ax.set_xlim(0,2)
    ax.set_xticks(ticks=np.arange(0, 2.4, 0.4))



def text_plot(axes):
    annotations = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axes):
        ax.text(0.94, 0.94, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))




def main():
    # Set the font properties
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    Layer1_Ali_Data=Read_Data('./ali0.8.txt')
    Layer1_Sarika_Data=Read_Data('./Saika0.8a.txt')
    Layer2=Read_Data('./Omega.txt')
    Number_of_Node = [i for i in range(1000)]
    axes=[plot_eshel(i) for i in range(1, 5)]
    plot_ran(axes[0],Number_of_Node,Layer1_Sarika_Data,Layer2)
    plot_ran_hist(axes[1],Layer1_Sarika_Data,Layer2)
    plot_reg(axes[2],Number_of_Node,Layer1_Ali_Data,Layer2)
    plot_ran_hist(axes[3],Layer1_Ali_Data,Layer2)
    text_plot(axes)


    x_lim=0.246
    plt.subplots_adjust(top = 0.97, bottom=0.08,left=x_lim,right=1-x_lim, hspace=0.3, wspace=0.3)
    plt.gcf().set_size_inches(28, 16)


    plt.savefig("Figure2_dpi100_1.jpg", dpi=100)
    plt.savefig("Figure2_dpi300.png", dpi=300)
    plt.savefig("Figure2_dpi300.jpg", dpi=300)
    plt.savefig("Figure2.pdf")
    with Image.open('Figure2_dpi300.png') as img:
        img.save('Figure2_dpi300.tiff', format='TIFF', compression='tiff_lzw')

    #plt.show()
    pass

if __name__=="__main__":
    main()