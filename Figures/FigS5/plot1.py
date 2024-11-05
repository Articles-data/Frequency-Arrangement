import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from scipy.signal import savgol_filter
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import AutoMinorLocator,MaxNLocator, MultipleLocator
import math

font1 = {'family': 'Times New Roman', 'color': '#000000', 'weight': 'normal'}
fonts = 'Times New Roman'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def plot_framework(number_of_pic):
        ax = plt.subplot(2,1,number_of_pic)
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=15)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
        return ax

def text_plot(axes):
        annotations = ['(a)', '(b)']
        for i, ax in enumerate(axes):
                ax.text(0.0206, 0.898, annotations[i], fontdict=font1, fontsize=30,
                        transform=ax.transAxes, ha='left', va='top',
                        bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))



def plot(ax,layer):
    Number_of_node=1000
    Number_of_step=40001#becuase start since 0
    address=r'./Data/Forward/'
    dw_left_cut=277
    dw_Right_cut=723
    Matrix_sor_0_2PI = [[0 for x in range(Number_of_node)] for y in range(Number_of_step)] 
    #Part syncroney
    Number_total=[0 for y in range(Number_of_step)]#total sync
    Total_sync=[0 for y in range(Number_of_step)]#total sync
    left_sync=[0 for y in range(Number_of_step)]
    mid_sync=[0 for y in range(Number_of_step)]
    right_sync=[0 for y in range(Number_of_step)]
    Degree=1.57 #degree alpha
    Degree_data='Degree_Radian='+str(Degree)#string degree
    copling=2.11 #coupling in layers
    file_name=Degree_data+'_copling='+str(copling)+f'layer{layer}(time)VS(Node)'
    data=np.loadtxt(address+'Save/Phases/'+file_name+'.txt')
    for timeforloop in range(0, Number_of_step):
        for y in range(1, Number_of_node+1):#for timeforloop step  
            Matrix_sor_0_2PI[timeforloop][y-1]=data[timeforloop][y]%(2*math.pi)
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


    if layer==1:
        colors = []
        line, = ax.plot(Number_total, savgol_filter(Total_sync, 80, 2),  color='#0000CD', label='$\mathrm{r}^\mathrm{I}$', linewidth = '2.8')#151575
        colors.append(plt.getp(line,'color'))
        line, = ax.plot(Number_total, savgol_filter(left_sync, 80, 2), color="#00CC00", label='$\mathrm{r}^\mathrm{I}_\mathrm{L}$', linewidth = '2.8')#298929
        colors.append(plt.getp(line,'color'))
        line, = ax.plot(Number_total, savgol_filter(mid_sync, 80, 2), color="#303030", label='$\mathrm{r}^\mathrm{I}_\mathrm{M}$', linewidth = '2.8')
        colors.append(plt.getp(line,'color'))
        line, = ax.plot(Number_total, savgol_filter(right_sync, 80, 2), color="#840000", label='$\mathrm{r}^\mathrm{I}_\mathrm{R}$', linewidth = '2.8')#751515
        colors.append(plt.getp(line,'color'))
        ax.set_xlabel('Time (t)', fontdict=font1, fontsize=30)
        ax.set_ylabel('Synchronization ($\mathrm{r}^\mathrm{I}$)', fontdict=font1, fontsize=30, labelpad=16)
        ax.set_xlim([100, 300])
        ax.set_ylim([0, 1.1])
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
        plt.setp(legend.texts, family='Times New Roman')
        plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=15)
        plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, pad=15)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
    elif layer==2:
        colors = []
        line, = ax.plot(Number_total, savgol_filter(Total_sync, 80, 2),  color='#0000CD', label='$\mathrm{r}^\mathrm{II}$', linewidth = '2.8')#151575
        colors.append(plt.getp(line,'color'))
        line, = ax.plot(Number_total, savgol_filter(left_sync, 80, 2), color="#00CC00", label='$\mathrm{r}^\mathrm{II}_\mathrm{L}$', linewidth = '2.8')#298929
        colors.append(plt.getp(line,'color'))
        line, = ax.plot(Number_total, savgol_filter(mid_sync, 80, 2), color="#303030", label='$\mathrm{r}^\mathrm{II}_\mathrm{M}$', linewidth = '2.8')
        colors.append(plt.getp(line,'color'))
        line, = ax.plot(Number_total, savgol_filter(right_sync, 80, 2), color="#840000", label='$\mathrm{r}^\mathrm{II}_\mathrm{R}$', linewidth = '2.8')#751515
        colors.append(plt.getp(line,'color'))
        ax.set_xlabel('Time (t)', fontdict=font1, fontsize=30)
        ax.set_ylabel('Synchronization ($\mathrm{r}^\mathrm{II}$)', fontdict=font1, fontsize=30, labelpad=16)
        ax.set_xlim([100, 300])
        ax.set_ylim([0, 1.1])
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
        plt.setp(legend.texts, family='Times New Roman')
        plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=15)
        plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, pad=15)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')





def main():
    fig = plt.figure()
    axes=[plot_framework(i) for i in range(1, 3)]

    Number_of_node=1000
    Number_of_step=40001#becuase start since 0
    address=r'./Data/Forward/'
    dw_left_cut=277
    dw_Right_cut=723
    Matrix_sor_0_2PI = [[0 for x in range(Number_of_node)] for y in range(Number_of_step)] 
    #Part syncroney
    Number_total=[0 for y in range(Number_of_step)]#total sync
    Total_sync=[0 for y in range(Number_of_step)]#total sync
    left_sync=[0 for y in range(Number_of_step)]
    mid_sync=[0 for y in range(Number_of_step)]
    right_sync=[0 for y in range(Number_of_step)]
    Degree=1.57 #degree alpha
    Degree_data='Degree_Radian='+str(Degree)#string degree
    copling=2.11 #coupling in layers
    file_name=Degree_data+'_copling='+str(copling)+'layer1(time)VS(Node)'
    data=np.loadtxt(address+'Save/Phases/'+file_name+'.txt')
    for timeforloop in range(0, Number_of_step):
        for y in range(1, Number_of_node+1):#for timeforloop step  
            Matrix_sor_0_2PI[timeforloop][y-1]=data[timeforloop][y]%(2*math.pi)
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
    colors = []
    line, = axes[0].plot(Number_total, savgol_filter(Total_sync, 80, 2),  color='#0000CD', label='$\mathrm{r}^\mathrm{I}$', linewidth = '2.8')#151575
    colors.append(plt.getp(line,'color'))
    line, = axes[0].plot(Number_total, savgol_filter(left_sync, 80, 2), color="#00CC00", label='$\mathrm{r}^\mathrm{I}_\mathrm{L}$', linewidth = '2.8')#298929
    colors.append(plt.getp(line,'color'))
    line, = axes[0].plot(Number_total, savgol_filter(mid_sync, 80, 2), color="#303030", label='$\mathrm{r}^\mathrm{I}_\mathrm{M}$', linewidth = '2.8')
    colors.append(plt.getp(line,'color'))
    line, = axes[0].plot(Number_total, savgol_filter(right_sync, 80, 2), color="#840000", label='$\mathrm{r}^\mathrm{I}_\mathrm{R}$', linewidth = '2.8')#751515
    colors.append(plt.getp(line,'color'))
    axes[0].set_xlabel('Time (t)', fontdict=font1, fontsize=30)
    axes[0].set_ylabel('Synchronization ($\mathrm{r}^\mathrm{I}$)', fontdict=font1, fontsize=30, labelpad=16)
    axes[0].set_xlim([100, 300])
    axes[0].set_ylim([0, 1.1])
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    axes[0].tick_params(axis='both', direction='in', which='major', length=22)
    axes[0].tick_params(axis='both', direction='in', which='minor', length=6)
    legend = axes[0].legend(loc=(1),fontsize=31)
    for color,text in zip(colors,legend.get_texts()):
        text.set_color(color)
        text.set_fontname(fonts)
    # Set the background color and transparency for the legend
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.94)  # Transparency level (0.3 for 30%)










    Number_of_node=1000
    Number_of_step=40001#becuase start since 0
    address=r'./Data/Forward/'
    dw_left_cut=277
    dw_Right_cut=723
    Matrix_sor_0_2PI = [[0 for x in range(Number_of_node)] for y in range(Number_of_step)] 
    #Part syncroney
    Number_total=[0 for y in range(Number_of_step)]#total sync
    Total_sync=[0 for y in range(Number_of_step)]#total sync
    left_sync=[0 for y in range(Number_of_step)]
    mid_sync=[0 for y in range(Number_of_step)]
    right_sync=[0 for y in range(Number_of_step)]
    Degree=1.57 #degree alpha
    Degree_data='Degree_Radian='+str(Degree)#string degree
    copling=2.11 #coupling in layers
    file_name=Degree_data+'_copling='+str(copling)+'layer2(time)VS(Node)'
    data=np.loadtxt(address+'Save/Phases/'+file_name+'.txt')
    for timeforloop in range(0, Number_of_step):
        for y in range(1, Number_of_node+1):#for timeforloop step  
            Matrix_sor_0_2PI[timeforloop][y-1]=data[timeforloop][y]%(2*math.pi)
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
    colors = []
    line, = axes[1].plot(Number_total, savgol_filter(Total_sync, 80, 2),  color='#0000CD', label='$\mathrm{r}^\mathrm{II}$', linewidth = '2.8')#151575
    colors.append(plt.getp(line,'color'))
    line, = axes[1].plot(Number_total, savgol_filter(left_sync, 80, 2), color="#00CC00", label='$\mathrm{r}^\mathrm{II}_\mathrm{L}$', linewidth = '2.8')#298929
    colors.append(plt.getp(line,'color'))
    line, = axes[1].plot(Number_total, savgol_filter(mid_sync, 80, 2), color="#303030", label='$\mathrm{r}^\mathrm{II}_\mathrm{M}$', linewidth = '2.8')
    colors.append(plt.getp(line,'color'))
    line, = axes[1].plot(Number_total, savgol_filter(right_sync, 80, 2), color="#840000", label='$\mathrm{r}^\mathrm{II}_\mathrm{R}$', linewidth = '2.8')#751515
    colors.append(plt.getp(line,'color'))
    axes[1].set_xlabel('Time (t)', fontdict=font1, fontsize=30)
    axes[1].set_ylabel('Synchronization ($\mathrm{r}^\mathrm{II}$)', fontdict=font1, fontsize=30, labelpad=16)
    axes[1].set_xlim([100, 300])
    axes[1].set_ylim([0, 1.1])
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    axes[1].tick_params(axis='both', direction='in', which='major', length=22)
    axes[1].tick_params(axis='both', direction='in', which='minor', length=6)
    legend = axes[1].legend(loc=(1),fontsize=31)
    for color,text in zip(colors,legend.get_texts()):
        text.set_color(color)
        text.set_fontname(fonts)
    # Set the background color and transparency for the legend
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.94)  # Transparency level (0.3 for 30%)











    '''axes[1].set_ylabel('Amplitude',  fontsize=31, labelpad=-12)
    axes[1].set_xlabel('Time (t)',  fontsize=31)#, labelpad=32)
    axes[1].set_xlim(0, 400)
    axes[1].set_ylim(-0.2, 0.2)
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    axes[1].tick_params(axis='both', direction='in', which='major', length=22)
    axes[1].tick_params(axis='both', direction='in', which='minor', length=6)'''
    





    text_plot(axes)
    
    y_lim=0.22
    x_lim=0.08
    plt.subplots_adjust(top = 1-y_lim, bottom=y_lim+0.02,left=x_lim,right=1-x_lim, hspace=0.4, wspace=0.3)
    plt.gcf().set_size_inches(28, 20)
    plt.savefig('./FigureS5-100.jpg', dpi=100)
    
    plt.savefig("FigureS6_dpi300.png", dpi=300)
    plt.savefig("FigureS6_dpi300.jpg", dpi=300)
    plt.savefig("FigureS6.pdf")
    with Image.open('FigureS6_dpi300.png') as img:
        img.save('FigureS6_dpi300.tiff', format='TIFF', compression='tiff_lzw')
pass

if __name__=="__main__":
    main()