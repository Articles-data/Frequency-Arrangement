import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

dataset_R = np.genfromtxt(fname='right_sync.txt',skip_header=1)
dataset_L = np.genfromtxt(fname='left_sync.txt',skip_header=1)

xsignalR = dataset_R[:,0]
ysignalR = dataset_R[:,1]
xsignalL = dataset_L[:,0]
ysignalL = dataset_L[:,1]

from scipy.signal import savgol_filter
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from scipy.signal import savgol_filter
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import AutoMinorLocator,MaxNLocator, MultipleLocator

in1=0.01
out1=0.045

in2=0.08
out2=0.18

in3=0.08
out3=0.18

font1 = {'family': 'Times New Roman', 'color': '#000000', 'weight': 'normal'}
fonts = 'Times New Roman'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def plot_framework(number_of_pic):
        ax = plt.subplot(3,1,number_of_pic)
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=15)
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=15)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
        return ax

def text_plot(axes):
        annotations = ['(a)', '(b)', '(c)']
        for i, ax in enumerate(axes):
                ax.text(0.026, 0.9, annotations[i], fontdict=font1, fontsize=30,
                        transform=ax.transAxes, ha='left', va='top',
                        bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))






time = np.linspace(0, 400,40000)


def main():
    fig = plt.figure()
    axes=[plot_framework(i) for i in range(1, 4)]

    
    colors = []
    
    
    
    
    line, = axes[0].plot(time, savgol_filter(ysignalR, 160, 2), color="#840000", label='$\mathrm{r}^\mathrm{II}_\mathrm{R}$',linewidth = '2.2')
    colors.append(plt.getp(line,'color'))
    line, = axes[0].plot(time, savgol_filter(ysignalL, 160, 2), color="#00CC00", label='$\mathrm{r}^\mathrm{II}_\mathrm{L}$',linewidth = '2.2')
    colors.append(plt.getp(line,'color'))
    
    legend = axes[0].legend(loc=(1),fontsize=31)
    
    for color,text in zip(colors,legend.get_texts()):
        text.set_color(color)
        text.set_fontname(fonts)
    # Set the background color and transparency for the legend
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.94)  # Transparency level (0.3 for 30%)
    
    
    
    axes[0].set_ylabel('Synchronization ($\mathrm{r}^\mathrm{II}$)',  fontsize=31)#, labelpad=37)
    axes[0].set_xlabel('Time (t)',  fontsize=31)
    axes[0].set_xlim(0, 400)
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())
    
    axes[0].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    
    axes[0].tick_params(axis='both', direction='in', which='major', length=22)
    axes[0].tick_params(axis='both', direction='in', which='minor', length=6)
    
    num_ticks = 6
    tick_locs = np.linspace(0, 1, num_ticks)
    axes[0].set_yticks(tick_locs)
    axes[0].set_ylim(0, 1)
    '''axes[0].tick_params(
        #axis='x',          # changes apply to the x-axis
        #which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        #top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off'''
    
    #axes[0].legend(loc="upper right",prop={'family': 'Times New Roman',  "size": 20 })
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    #______________________________________________________filter L     
    # define a frequency range to filter out
    freq = np.fft.fftfreq(len(savgol_filter(ysignalL, 160, 2)), time[1]-time[0])
    ft = np.fft.fft(savgol_filter(ysignalL, 160, 2))
    
    freq_min_delta = in3  # Hz
    freq_max_delta = out3  # Hz
    # create a filter function that sets the Fourier coefficients
    # outside the frequency range to zero
    filter_delta = np.logical_or(freq < freq_min_delta, freq > freq_max_delta)
    fft_vals_filtered_delta = ft.copy()
    fft_vals_filtered_delta[filter_delta] = 0
    # compute the inverse Fourier transform of the filtered coefficients
    signal_filtered_deltaL = np.real(np.fft.ifft(fft_vals_filtered_delta)) 
    fft_vals = np.fft.fft(ysignalL)
    sampling_freq = 1 / (xsignalL[1] - xsignalL[0])  # Assuming evenly spaced samples
    freqs = np.fft.fftfreq(len(ysignalL), 1 / sampling_freq)
    #plt.plot(freqs,  savgol_filter(np.abs(fft_vals), 80, 2), label='Fourier Transform',color='r')
    fft_vals[0]=0
    # plot the original and filtered signals
    
    
    
    # Define the y-limits for filling
    y_bottom = axes[1].get_ylim()[0]  # Bottom of the y-axis
    y_top = axes[1].get_ylim()[1]     # Top of the y-axis
    
    freq = np.fft.fftfreq(len(savgol_filter(ysignalR, 160, 2)), time[1]-time[0])
    ft = np.fft.fft(savgol_filter(ysignalR, 160, 2))
    freq_min_delta = in3  # Hz
    freq_max_delta = out3  # Hz
    # create a filter function that sets the Fourier coefficients
    # outside the frequency range to zero
    filter_delta = np.logical_or(freq < freq_min_delta, freq > freq_max_delta)
    fft_vals_filtered_delta = ft.copy()
    fft_vals_filtered_delta[filter_delta] = 0
    # compute the inverse Fourier transform of the filtered coefficients
    signal_filtered_deltaR = np.real(np.fft.ifft(fft_vals_filtered_delta))
    # plot the original and filtered signals
    
    fft_vals = np.fft.fft(ysignalR)
    sampling_freq = 1 / (xsignalR[1] - xsignalR[0])  # Assuming evenly spaced samples
    freqs = np.fft.fftfreq(len(ysignalR), 1 / sampling_freq)
    #plt.plot(freqs,  savgol_filter(np.abs(fft_vals), 80, 2), label='Fourier Transform',color='r')
    fft_vals[0]=0
    
    #axes[1].legend(loc="upper right",prop={'family': 'Times New Roman',  "size": 20 })
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    phase_A = np.angle(hilbert(signal_filtered_deltaL))
    
    amps_B = np.abs(hilbert(signal_filtered_deltaL))#SATR
    amps_Br = np.abs(hilbert(signal_filtered_deltaR))#SATR
    
    freq = np.fft.fftfreq(len(savgol_filter(ysignalR, 160, 2)), time[1]-time[0])
    ft = np.fft.fft(savgol_filter(ysignalR, 160, 2))
    freq_min_delta = in1  # Hz
    freq_max_delta = out1  # Hz
    # create a filter function that sets the Fourier coefficients
    # outside the frequency range to zero
    filter_delta = np.logical_or(freq < freq_min_delta, freq > freq_max_delta)
    fft_vals_filtered_delta = ft.copy()
    fft_vals_filtered_delta[filter_delta] = 0
    # compute the inverse Fourier transform of the filtered coefficients
    signal_filtered_deltaR2 = np.real(np.fft.ifft(fft_vals_filtered_delta))
    
    
    
    
    freq = np.fft.fftfreq(len(savgol_filter(ysignalL, 160, 2)), time[1]-time[0])
    ft = np.fft.fft(savgol_filter(ysignalL, 160, 2))
    freq_min_delta = in1  # Hz
    freq_max_delta = out1  # Hz
    # create a filter function that sets the Fourier coefficients
    # outside the frequency range to zero
    filter_delta = np.logical_or(freq < freq_min_delta, freq > freq_max_delta)
    fft_vals_filtered_delta = ft.copy()
    fft_vals_filtered_delta[filter_delta] = 0
    # compute the inverse Fourier transform of the filtered coefficients
    signal_filtered_deltaL2 = np.real(np.fft.ifft(fft_vals_filtered_delta))
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    phase_R = np.angle(hilbert(savgol_filter(signal_filtered_deltaR, 160, 2)))
    phase_L = np.angle(hilbert(savgol_filter(signal_filtered_deltaL, 160, 2)))
    
    
    
    #axes[2].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False,  labelsize=22, labelcolor='#262626')
    #axes[2].tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False,  labelsize=22, labelcolor='#262626')
    colors = []
    S2R=signal_filtered_deltaR
    line, = axes[2].plot(time, signal_filtered_deltaR, color="#840000", label='$\mathrm{r}^\mathrm{II}_\mathrm{R}$ (MFO)',alpha=1,linewidth = '2.8')
    axes[2].fill_between(time, signal_filtered_deltaR,where=(signal_filtered_deltaR>0),color="#840000",alpha=0.2)
    axes[2].fill_between(time, signal_filtered_deltaR,where=(signal_filtered_deltaR<0),color="#840000",alpha=0.2)
    colors.append(plt.getp(line,'color'))
    S2L=signal_filtered_deltaL
    
    line, = axes[2].plot(time, signal_filtered_deltaL, color="#00CC00", label='$\mathrm{r}^\mathrm{II}_\mathrm{L}$ (MFO)',alpha=1,linewidth = '2.8')
    axes[2].fill_between(time, signal_filtered_deltaL,where=(signal_filtered_deltaL>0),color="#00CC00",alpha=0.2)
    axes[2].fill_between(time, signal_filtered_deltaL,where=(signal_filtered_deltaL<0),color="#00CC00",alpha=0.2)
    colors.append(plt.getp(line,'color'))
    axes[2].xaxis.set_minor_locator(AutoMinorLocator())
    axes[2].yaxis.set_minor_locator(AutoMinorLocator())
    
    axes[2].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    
    axes[2].tick_params(axis='both', direction='in', which='major', length=22)
    axes[2].tick_params(axis='both', direction='in', which='minor', length=6)
    avCos=0
    for i in range (len(phase_R)):
        avCos+=np.cos(phase_R[i]-phase_L[i])
    print(avCos/len(phase_R))
    
    legend = axes[2].legend(loc=(1),fontsize=31)
    
    for color,text in zip(colors,legend.get_texts()):
        text.set_color(color)
        text.set_fontname(fonts)
    # Set the background color and transparency for the legend
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.94)  # Transparency level (0.3 for 30%)
    
    
    
    
    
    
    
    
    
    num_ticks = 5
    tick_locs = np.linspace(-0.2, 0.2, num_ticks)
    axes[2].set_yticks(tick_locs)
    
    
    axes[2].set_ylabel('Amplitude',  fontsize=31, labelpad=-12)
    axes[2].set_xlabel('Time (t)',  fontsize=31)#, labelpad=32)
    axes[2].set_xlim(0, 400)
    axes[2].set_ylim(-0.2, 0.2)
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    phase_R = np.angle(hilbert(savgol_filter(signal_filtered_deltaR2, 160, 2)))
    phase_L = np.angle(hilbert(savgol_filter(signal_filtered_deltaL2, 160, 2)))
    
    amps_B = np.abs(hilbert(signal_filtered_deltaL2))#SATR
    amps_Br = np.abs(hilbert(signal_filtered_deltaR2))#SATR
    colors = []
    
    
    
    #axes[1].tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False,  labelsize=22, labelcolor='#262626')
    #axes[1].tick_params(axis='x', which='both', left=True, right=False, labelleft=True, labelright=False,  labelsize=22, labelcolor='#262626')
    S1R=signal_filtered_deltaR2
    line, = axes[1].plot(time, signal_filtered_deltaR2, color="#840000", label='$\mathrm{r}^\mathrm{II}_\mathrm{R}$ (LFO)',alpha=1,linewidth = '2.8')
    axes[1].fill_between(time, signal_filtered_deltaR2,where=(signal_filtered_deltaR2>0),color="#840000",alpha=0.3)
    axes[1].fill_between(time, signal_filtered_deltaR2,where=(signal_filtered_deltaR2<0),color="#840000",alpha=0.3)
    S1L=signal_filtered_deltaL2
    colors.append(plt.getp(line,'color'))
    line, = axes[1].plot(time, signal_filtered_deltaL2, color="#00CC00", label='$\mathrm{r}^\mathrm{II}_\mathrm{L}$ (LFO)',alpha=1,linewidth = '2.8')
    axes[1].fill_between(time, signal_filtered_deltaL2,where=(signal_filtered_deltaL2>0),color="#00CC00",alpha=0.2)
    axes[1].fill_between(time, signal_filtered_deltaL2,where=(signal_filtered_deltaL2<0),color="#00CC00",alpha=0.2)
    colors.append(plt.getp(line,'color'))
    
    
    avCos=0
    for i in range (len(phase_R)):
        avCos+=np.cos(phase_R[i]-phase_L[i])
    print(avCos/len(phase_R))
    
    
    legend = axes[1].legend(loc=(1),fontsize=31)
    
    for color,text in zip(colors,legend.get_texts()):
        text.set_color(color)
        text.set_fontname(fonts)
    # Set the background color and transparency for the legend
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.94)  # Transparency level (0.3 for 30%)
    
    
    
    
    
    
    
    
    
    
    num_ticks = 5
    tick_locs = np.linspace(-0.2, 0.2, num_ticks)
    axes[1].set_yticks(tick_locs)
    axes[1].set_ylabel('Amplitude',  fontsize=31, labelpad=-12)
    axes[1].set_xlabel('Time (t)',  fontsize=31)#, labelpad=32)
    
    axes[1].set_xlim(0, 400)
    axes[1].set_ylim(-0.2, 0.2)
    
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    
    axes[1].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    
    axes[1].tick_params(axis='both', direction='in', which='major', length=22)
    axes[1].tick_params(axis='both', direction='in', which='minor', length=6)
    
    
    text_plot(axes)
    
    y_lim=0.06
    x_lim=0.08
    plt.subplots_adjust(top = 1-y_lim, bottom=y_lim+0.02,left=x_lim,right=1-x_lim, hspace=0.4, wspace=0.3)
    plt.gcf().set_size_inches(28, 20)
    

    plt.savefig("FigureS6_dpi100_1.jpg", dpi=100)
    plt.savefig("FigureS6_dpi300.png", dpi=300)
    plt.savefig("FigureS6_dpi300.jpg", dpi=300)
    plt.savefig("FigureS6.pdf")
    with Image.open('FigureS6_dpi300.png') as img:
        img.save('FigureS6_dpi300.tiff', format='TIFF', compression='tiff_lzw')
pass

if __name__=="__main__":
    main()