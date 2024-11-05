import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os        
from matplotlib.ticker import AutoMinorLocator,MaxNLocator, MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from scipy.signal import savgol_filter

font1 = {'family': 'Times New Roman', 'color': '#000000', 'weight': 'normal'}
fonts = 'Times New Roman'

def plot(ax):
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
    ax.set_xlabel('Node', fontdict=font1, fontsize=31)
    ax.set_ylabel('Node', fontdict=font1, fontsize=31, labelpad=-10)
    ax.set_ylim(0,1000)
    ax.set_xlim(0,1000)



def text_plot(axes):
    annotations = ['(a)', '(b)', '(c)', '(d)','(e)', '(f)', '(g)', '(h)', '(i)']
    for i, ax in enumerate(axes):
        if i<3:
            ax.text(0.02, 0.89, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))
        elif i==3 or i==6:
            ax.text(0.042, 0.89, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))
        elif i==4 or i==7:
            ax.text(0.1, 0.89, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))
        else:
            ax.text(-0.14, 0.89, annotations[i], fontdict=font1, fontsize=30,
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle="round", ec="#000000", fc=(1, 1, 1, 0.80)))

def plot_framework(number_of_pic):
    if number_of_pic==1:
        ax = plt.subplot(5,4,(1,4))
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=10,  labelsize=32, labelcolor='#000000')
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=5,  labelsize=32, labelcolor='#000000')
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
        plt.ylim(0,1)
        plt.xlim(0,400)
        plt.ylabel('Synchronization ($\mathrm{r}^\mathrm{II}_\mathrm{R}$)', fontdict=font1, fontsize=38, labelpad=34)
        plt.xlabel('Time (t)',fontdict=font1,  fontsize=38)
    if number_of_pic==2:
        ax = plt.subplot(5,4,(5,8))
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=10,  labelsize=32, labelcolor='#000000')
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=5,  labelsize=32, labelcolor='#000000')
        plt.ylabel('FFT', fontdict=font1, fontsize=38, labelpad=18)
        plt.xlabel('Frequency (Hz)', fontdict=font1, fontsize=38)
        plt.ylim(0, 2000)
        plt.xlim(0, 4)
        num_ticks = 5
        tick_locs = np.linspace(0, 2000, num_ticks)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(tick_locs,font=fonts,fontsize=31,color= '#000000')
    if number_of_pic==3:
        ax = plt.subplot(5,4,(9,12))
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=10,  labelsize=32, labelcolor='#000000')
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=5,  labelsize=32, labelcolor='#000000')
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
        plt.ylabel('Amplitude',fontdict=font1,  fontsize=38, labelpad=18)
        plt.xlabel('Time (t)',fontdict=font1,  fontsize=38)
        plt.xlim(0, 400)
        plt.ylim(-0.2, 0.2)




    if number_of_pic==4:
        ax = plt.subplot(5,4,(13,14))
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=10,  labelsize=32, labelcolor='#000000')
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=5,  labelsize=32, labelcolor='#000000')
        plt.xlim(0, 400)
        plt.ylim(-0.2, 0.2)
        plt.ylabel('Amplitude',fontdict=font1,  fontsize=38, labelpad=18)
        plt.xlabel('Time (t)',fontdict=font1,  fontsize=38, labelpad=20)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        num_ticks = 5
        tick_locs = np.linspace(-0.2, 0.2, num_ticks)
        plt.yticks(tick_locs,font=fonts,fontsize=31,color= '#000000')
    if number_of_pic==5:
        ax = plt.subplot(5,4,(15))
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=10,  labelsize=32, labelcolor='#000000')
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=5,  labelsize=32, labelcolor='#000000')
        plt.ylim(0, 0.1)
        plt.xlim(-np.pi-0.19, np.pi+0.19)
        plt.yticks(font=fonts,fontsize=31,color= '#000000')
        num_ticks = 3
        tick_locs = np.linspace(-3.14, 3.14, num_ticks)
        plt.xticks(tick_locs,font=fonts,fontsize=31,color= '#000000')
        plt.ylabel('Amplitude',fontdict=font1,  fontsize=38)
        plt.xlabel('Phase',fontdict=font1,  fontsize=38, labelpad=20)
    if number_of_pic==6:
        ax = plt.subplot(5,4,(16), projection='polar')
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=20,  labelsize=32, labelcolor='#000000')
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks([])
    if number_of_pic==7:
        ax = plt.subplot(5,4,(17,18))
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=10,  labelsize=32, labelcolor='#000000')
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=5,  labelsize=32, labelcolor='#000000')
        plt.xlim(0, 400)
        plt.ylim(-0.2, 0.2)
        plt.ylabel('Amplitude',fontdict=font1,  fontsize=38, labelpad=18)
        plt.xlabel('Time (t)',fontdict=font1,  fontsize=38, labelpad=20)
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        num_ticks = 5
        tick_locs = np.linspace(-0.2, 0.2, num_ticks)
        plt.yticks(tick_locs,font=fonts,fontsize=31,color= '#000000')
    if number_of_pic==8:
        ax = plt.subplot(5,4,(19))
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=10,  labelsize=32, labelcolor='#000000')
        plt.tick_params(axis='y', which='both' , left=True, right=True, labelbottom=True, labeltop=False, pad=5,  labelsize=32, labelcolor='#000000')
        plt.ylim(0, 0.1)
        plt.xlim(-np.pi-0.19, np.pi+0.19)
        num_ticks = 3
        tick_locs = np.linspace(-3.14, 3.14, num_ticks)
        plt.xticks(tick_locs,font=fonts,fontsize=31,color= '#000000')
        plt.ylabel('Amplitude',fontdict=font1,  fontsize=38)
        plt.xlabel('Phase',fontdict=font1,  fontsize=38, labelpad=20)
    if number_of_pic==9:
        ax = plt.subplot(5,4,(20), projection='polar')
        plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, pad=20,  labelsize=32, labelcolor='#000000')
        plt.xticks(font=fonts,fontsize=31,color= '#000000')
        plt.yticks([])
    return ax

in1=0.01
out1=0.045
in2=0.08
out2=0.18
in3=0.5
out3=3.5
font1 = {'family': 'Times New Roman', 'color': '#000000', 'weight': 'normal'}
fonts = 'Times New Roman'
dataset = np.genfromtxt(fname='right_sync.txt',skip_header=1)
xsignal = dataset[:,0]
ysignal = dataset[:,1]
signal_shift=ysignal
time = np.linspace(0, 400,40000)
freq = np.fft.fftfreq(len(signal_shift), time[1]-time[0])
ft = np.fft.fft(signal_shift)
freq=freq#*1.89
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'



fig = plt.figure()
axes=[plot_framework(i) for i in range(1, 10)]


#((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((________________________ 1.signal
axes[0].plot(xsignal, signal_shift, color="#840000",linewidth = '2.8')
axes[0].xaxis.set_minor_locator(AutoMinorLocator())
axes[0].yaxis.set_minor_locator(AutoMinorLocator())
axes[0].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
axes[0].tick_params(axis='both', direction='in', which='major', length=22)
axes[0].tick_params(axis='both', direction='in', which='minor', length=6)
#((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((________________________ 2.fft
fft_vals = np.fft.fft(ysignal)
sampling_freq = 1 / (xsignal[1] - xsignal[0])  # Assuming evenly spaced samples
freqs = np.fft.fftfreq(len(ysignal), 1 / sampling_freq)
fft_vals[0]=0
axes[1].plot(freqs,  savgol_filter(np.abs(fft_vals), 15, 2), label='Filtered Fourier Transform',color='#840000',linewidth = '4.4')
axes[1].fill(freqs,  savgol_filter(np.abs(fft_vals), 15, 2),color='#840000',alpha=0.8)
axes[1].fill_betweenx([-100, 100000], in1, out1, color="#3C3C3C", alpha=0.5)
axes[1].fill_betweenx([-100, 100000], in2, out2, color="#007600", alpha=0.5)
axes[1].fill_betweenx([-100, 100000], in3, out3, color="#157DEC", alpha=0.5)
axes[1].axvline(x=in1, color="#000000", alpha=0.7)
axes[1].axvline(x=(in1+out1)/2, color="#000000", linestyle='--', alpha=0.7)
axes[1].axvline(x=out1, color="#000000", alpha=0.7)
axes[1].axvline(x=in2, color="#003A00", alpha=0.7)
axes[1].axvline(x=(in2+out2)/2, color="#003A00", linestyle='--', alpha=0.7)
axes[1].axvline(x=out2, color="#003A00", alpha=0.7)
axes[1].axvline(x=in3, color="#002392", alpha=0.7)
axes[1].axvline(x=(in3+out3)/2, color="#002392", linestyle='--', alpha=0.7)
axes[1].axvline(x=out3, color="#002392", alpha=0.7)
#((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((________________________ 3.
freq_min_g = in1  # Hz
freq_max_g = out1  # Hz
filter_g = np.logical_or(freqs < freq_min_g, freqs > freq_max_g)
fft_vals_filtered_g = fft_vals.copy()
fft_vals_filtered_g[filter_g] = 0
signal_filtered_g = np.real(np.fft.ifft(fft_vals_filtered_g))
line, =axes[2].plot(xsignal, signal_filtered_g, color='#000000',linewidth = '4.4')
from scipy.signal import hilbert, chirp
phase_A = np.angle(hilbert(signal_filtered_g))
amps_B = np.abs(hilbert(signal_filtered_g))#SATR
ffp = phase_A/20
AfA = amps_B
colors = []
axes[2].plot(xsignal, ffp, color='#C0BDBE',linewidth = '4.4')
line, =axes[2].plot(xsignal, ffp, color='#FF5500', label='Phase', linestyle='dashed',linewidth = '4.4')
colors.append(plt.getp(line,'color'))
axes[2].plot(xsignal, AfA, color='#C0BDBE',linewidth = '4.4')
line, =axes[2].plot(xsignal, AfA, color='#800080', label='Amplitude', linestyle='dashed',linewidth = '4.4')
colors.append(plt.getp(line,'color'))
legend = axes[2].legend(loc=(0.844,0.617),fontsize=32)
for color,text in zip(colors,legend.get_texts()):
    text.set_color(color)
    text.set_fontname(fonts)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.94)  # Transparency level (0.3 for 30%)

axes[2].xaxis.set_minor_locator(AutoMinorLocator())
axes[2].yaxis.set_minor_locator(AutoMinorLocator())
axes[2].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
axes[2].tick_params(axis='both', direction='in', which='major', length=22)
axes[2].tick_params(axis='both', direction='in', which='minor', length=6)

#((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((________________________ 6.top polar
import cmath
#______________________________________________________filter      _A
freq_min_A = in1  # Hz
freq_max_A = out1  # Hz
filter_A = np.logical_or(freq < freq_min_A, freq > freq_max_A)
fft_vals_filtered_A = ft.copy()
fft_vals_filtered_A[filter_A] = 0
signal_filtered_A = np.real(np.fft.ifft(fft_vals_filtered_A))
#______________________________________________________filter      _B
freq_min_B = in2  # Hz0.06, 0.15
freq_max_B = out2 # Hz
filter_B = np.logical_or(freq < freq_min_B, freq > freq_max_B)
fft_vals_filtered_B = ft.copy()
fft_vals_filtered_B[filter_B] = 0
signal_filtered_B = np.real(np.fft.ifft(fft_vals_filtered_B))
phase_A = np.angle(hilbert(signal_filtered_A))
amps_B = np.abs(hilbert(signal_filtered_B))#SATR
phase_delta = np.angle(hilbert(signal_filtered_A))
amps_gamma = np.abs(hilbert(signal_filtered_B))#SATR
phase_amps_gamma= np.angle(hilbert(amps_gamma))
defrent_each_angles=phase_delta-phase_amps_gamma
#calculate Average radius
def order_parameter(N, phi):
    rc = 0.0
    rs = 0.0
    for j in range(N):
        rc += math.cos(phi[j])
        rs += math.sin(phi[j])
    return math.sqrt(rc**2 + rs**2) / (1.0 * N)
average_radius2=order_parameter(len(defrent_each_angles),defrent_each_angles)
#calculate Average angels
resolt_rad=0
average_angels=0
for i in range(0,len(defrent_each_angles)):
    resolt_rad=resolt_rad+cmath.exp(complex(0, defrent_each_angles[i]))
resolt_rad=resolt_rad/len(defrent_each_angles)
if resolt_rad.real<0:
    resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
    average_angels=resolt_rad.real+math.pi
else:
    resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
    average_angels=resolt_rad.real
print("PLV out2=")
print(average_radius2)
axes[5].plot([defrent_each_angles,defrent_each_angles], [0,1],alpha=0.005, color='k')
axes[5].plot([average_angels,average_angels], [0,np.abs(average_radius2)],alpha=1, color='r', linewidth = '4')    
#((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((________________________ 5.top bins
from scipy import stats
#______________________________________________________filter      _A
freq_min_A = in1  # Hz
freq_max_A = out1  # Hz
filter_A = np.logical_or(freq < freq_min_A, freq > freq_max_A)
fft_vals_filtered_A = ft.copy()
fft_vals_filtered_A[filter_A] = 0
signal_filtered_A = np.real(np.fft.ifft(fft_vals_filtered_A))
#______________________________________________________filter      _B
freq_min_B = in2  # Hz
freq_max_B = out2 # Hz
filter_B = np.logical_or(freq < freq_min_B, freq > freq_max_B)
fft_vals_filtered_B = ft.copy()
fft_vals_filtered_B[filter_B] = 0
signal_filtered_B = np.real(np.fft.ifft(fft_vals_filtered_B))
phase_A = np.angle(hilbert(signal_filtered_A))
amps_B = np.abs(hilbert(signal_filtered_B))#SATR
phase_bins = np.linspace(-np.pi, np.pi+0.36959914, 19) # example number of bins
kAfAlffp, _, _ = stats.binned_statistic(ffp, AfA, bins=phase_bins, statistic='mean')
n, bins = np.histogram(phase_A, bins=18, range=(-np.pi,np.pi), weights=amps_B, density=False)
P = kAfAlffp / np.sum(kAfAlffp)
P2 = n / np.sum(n)
MI_2=0
for i in range (0,len(P2)):
    if P2[i]!=0:
        MI_2 = MI_2+(P2[i] * np.log10(P2[i]))
    else:
        MI_2 = MI_2
MI_2=1+(MI_2 / np.log10(len(P2)))
print("MI out2=")
print(MI_2)
axes[4].bar(phase_bins[:-1], P2,edgecolor='k', width=.366, color = u'#7C7C7C')
axes[4].tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False)

'''axes[4].tick_params(
    bottom=False,      # ticks along the bottom edge are off
    labelbottom=False) # labels along the bottom edge are off'''
#((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((________________________ 4.top signal
freq_min_a = in1  # Hz
freq_max_a = out1  # Hz
filter_a = np.logical_or(freqs < freq_min_a, freqs > freq_max_a)
fft_vals_filtered_a = fft_vals.copy()
fft_vals_filtered_a[filter_a] = 0
signal_filtered_a = np.real(np.fft.ifft(fft_vals_filtered_a))
freq_min_g = in2  # Hz
freq_max_g = out2 # Hz
filter_g = np.logical_or(freqs < freq_min_g, freqs > freq_max_g)
fft_vals_filtered_g = fft_vals.copy()
fft_vals_filtered_g[filter_g] = 0
signal_filtered_g = np.real(np.fft.ifft(fft_vals_filtered_g))
phase_A = np.angle(hilbert(signal_filtered_g))
amps_B = np.abs(hilbert(signal_filtered_g))#SATR
ffp = phase_A/20
AfA = amps_B
colors = []
line, = axes[3].plot(xsignal, signal_filtered_g, color='#157DEC', label='HFO (0.5-3.5 Hz)',linewidth = '4')
colors.append(plt.getp(line,'color'))
line, =axes[3].plot(xsignal, signal_filtered_g, color= '#00AF00', label='MFO (0.08-0.18 Hz)',linewidth = '4')
colors.append(plt.getp(line,'color'))
line, =axes[3].plot(xsignal, signal_filtered_a, color='#000000', label='LFO (0.01-0.045 Hz)',linewidth = '4')
colors.append(plt.getp(line,'color'))
legend = axes[3].legend(loc=(1.654,3.259),fontsize=32)
for color,text in zip(colors,legend.get_texts()):
    text.set_color(color)
    text.set_fontname(fonts)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.94)  # Transparency level (0.3 for 30%)
num_ticks = 5
tick_locs = np.linspace(-0.2, 0.2, num_ticks)
axes[3].xaxis.set_minor_locator(AutoMinorLocator())
axes[3].yaxis.set_minor_locator(AutoMinorLocator())
axes[3].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
axes[3].tick_params(axis='both', direction='in', which='major', length=22)
axes[3].tick_params(axis='both', direction='in', which='minor', length=6)
#((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((________________________ 9.down polar
#______________________________________________________filter      _A
freq_min_A = in1  # Hz
freq_max_A = out1  # Hz
filter_A = np.logical_or(freq < freq_min_A, freq > freq_max_A)
fft_vals_filtered_A = ft.copy()
fft_vals_filtered_A[filter_A] = 0
signal_filtered_A = np.real(np.fft.ifft(fft_vals_filtered_A))
#______________________________________________________filter      _B
freq_min_B = in3  # Hz
freq_max_B = out3 # Hz
filter_B = np.logical_or(freq < freq_min_B, freq > freq_max_B)
fft_vals_filtered_B = ft.copy()
fft_vals_filtered_B[filter_B] = 0
signal_filtered_B = np.real(np.fft.ifft(fft_vals_filtered_B))
phase_A = np.angle(hilbert(signal_filtered_A))
amps_B = np.abs(hilbert(signal_filtered_B))#SATR
phase_delta = np.angle(hilbert(signal_filtered_A))
amps_gamma = np.abs(hilbert(signal_filtered_B))#SATR
phase_amps_gamma= np.angle(hilbert(amps_gamma))
defrent_each_angles=phase_delta-phase_amps_gamma
#calculate Average radius
def order_parameter(N, phi):
    rc = 0.0
    rs = 0.0
    for j in range(N):
        rc += math.cos(phi[j])
        rs += math.sin(phi[j])
    return math.sqrt(rc**2 + rs**2) / (1.0 * N)
average_radius=order_parameter(len(defrent_each_angles),defrent_each_angles)
print("PLV out3=")
print(average_radius)
#calculate Average angels
resolt_rad=0
average_angels=0
for i in range(0,len(defrent_each_angles)):
    resolt_rad=resolt_rad+cmath.exp(complex(0, defrent_each_angles[i]))
resolt_rad=resolt_rad/len(defrent_each_angles)
if resolt_rad.real<0:
    resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
    average_angels=resolt_rad.real+math.pi
else:
    resolt_rad=cmath.atan(resolt_rad.imag/resolt_rad.real)#resolt_degree=resolt_rad*180/math.pi
    average_angels=resolt_rad.real
axes[8].plot([defrent_each_angles,defrent_each_angles], [0,1],alpha=0.005, color='k')
axes[8].plot([average_angels,average_angels], [0,np.abs(average_radius)],alpha=1, color='r', linewidth = '4')
#((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((________________________ 8.down bins
#______________________________________________________filter      _A
freq_min_A = in1  # Hz
freq_max_A = out1  # Hz
filter_A = np.logical_or(freq < freq_min_A, freq > freq_max_A)
fft_vals_filtered_A = ft.copy()
fft_vals_filtered_A[filter_A] = 0
signal_filtered_A = np.real(np.fft.ifft(fft_vals_filtered_A))
#______________________________________________________filter      _B
freq_min_B = in3  # Hz
freq_max_B = out3 # Hz
filter_B = np.logical_or(freq < freq_min_B, freq > freq_max_B)
fft_vals_filtered_B = ft.copy()
fft_vals_filtered_B[filter_B] = 0
signal_filtered_B = np.real(np.fft.ifft(fft_vals_filtered_B))
phase_A = np.angle(hilbert(signal_filtered_A))
amps_B = np.abs(hilbert(signal_filtered_B))#SATR
# Bin the phases and calculate the mean amplitude for each bin
phase_bins = np.linspace(-np.pi, np.pi+0.36959914, 19) # example number of bins
kAfAlffp, _, _ = stats.binned_statistic(ffp, AfA, bins=phase_bins, statistic='mean')
n, bins = np.histogram(phase_A, bins=18, range=(-np.pi,np.pi), weights=amps_B, density=False)
P = kAfAlffp / np.sum(kAfAlffp)
P2 = n / np.sum(n)
'''axes[7].tick_params(
    #axis='x',          # changes apply to the x-axis
    #which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    #top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off'''
axes[7].tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False)

# Calculate the MI
MI_1=0
for i in range (0,len(P2)):
    if P2[i]!=0:
        MI_1 = MI_1+(P2[i] * np.log10(P2[i]))
    else:
        MI_1 = MI_1
MI_1=1+(MI_1 / np.log10(len(P2)))
print("MI out3=")
print(MI_1)
axes[7].bar(phase_bins[:-1], P2,edgecolor='k', width=.366, color = '#7C7C7C')
#((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((________________________ 7.down signal
freq_min_a = in1  # Hz
freq_max_a = out1  # Hz
filter_a = np.logical_or(freqs < freq_min_a, freqs > freq_max_a)
fft_vals_filtered_a = fft_vals.copy()
fft_vals_filtered_a[filter_a] = 0
signal_filtered_a = np.real(np.fft.ifft(fft_vals_filtered_a))
freq_min_b = in3  # Hz
freq_max_b = out3 # Hz
filter_b = np.logical_or(freqs < freq_min_b, freqs > freq_max_b)
fft_vals_filtered_b = fft_vals.copy()
fft_vals_filtered_b[filter_b] = 0
signal_filtered_b = np.real(np.fft.ifft(fft_vals_filtered_b))
phase_A = np.angle(hilbert(signal_filtered_b))
amps_B = np.abs(hilbert(signal_filtered_b))#SATR
ffp = phase_A/20
AfA = amps_B
colors = []
line, = axes[6].plot(xsignal, signal_filtered_b, color='#157DEC', label='0.5-3.5 Hz',linewidth = '4')
colors.append(plt.getp(line,'color'))
line, = axes[6].plot(xsignal, signal_filtered_a, label='0.01-0.045 Hz', color='#000000',linewidth = '4')
colors.append(plt.getp(line,'color'))
axes[6].xaxis.set_minor_locator(AutoMinorLocator())
axes[6].yaxis.set_minor_locator(AutoMinorLocator())
axes[6].tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
axes[6].tick_params(axis='both', direction='in', which='major', length=22)
axes[6].tick_params(axis='both', direction='in', which='minor', length=6)






















text_plot(axes)



y_lim=0.048
x_lim=0.06
plt.subplots_adjust(top = 1-y_lim+0.02, bottom=y_lim,left=x_lim+0.02,right=1-x_lim+0.02, hspace=0.4, wspace=0.38)


plt.gcf().set_size_inches(28, 32)

plt.savefig("Figure7_100.jpg", dpi=100)
plt.savefig("Figure7_dpi300.png", dpi=300)
plt.savefig("Figure7_dpi300.jpg", dpi=300)
plt.savefig("Figure7.pdf")
with Image.open('Figure7_dpi300.png') as img:
    img.save('Figure7_dpi300.tiff', format='TIFF', compression='tiff_lzw')

