import numpy as np
import pandas as pd
from scipy.special import erfinv, erf
from scipy.signal import deconvolve, convolve, resample, decimate, resample_poly
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, ifftshift
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 6})

def deconvolve_fun(obs, signal):
    """Find convolution filter
    
    Finds convolution filter from observation and impulse response.
    Noise-free signal is assumed.
    """
    signal = np.hstack((signal, np.zeros(len(obs) - len(signal)))) 
    Fobs = np.fft.fft(obs)
    Fsignal = np.fft.fft(signal)
    filt = np.fft.ifft(Fobs/Fsignal)
    return filt
    
def wiener_deconvolution(signal, kernel, lambd = 1e-3):
    """Applies Wiener deconvolution to find true observation from signal and filter
    
    The function can be also used to estimate filter from true signal and observation
    """
    # zero pad the kernel to same length
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) 
    H = fft(kernel)
    deconvolved = np.real(ifft(fft(signal)*np.conj(H)/(H*np.conj(H) + lambd**2)))
    return deconvolved
    
def get_signal(time, offset_x, offset_y, reps = 4, lambd = 1e-3):
    """Model step response as error function
    """
    ramp_up = erf(time * multiplier)
    ramp_down = 1 - ramp_up
    if (reps % 1) == 0.5:
        signal =  np.hstack(( np.zeros(offset_x), 
                                ramp_up)) + offset_y
    else:
        signal =  np.hstack(( np.zeros(offset_x),
                                np.tile(np.hstack((ramp_up, ramp_down)), reps),
                                np.zeros(offset_x))) + offset_y 
      
    signal += np.random.randn(*signal.shape) * lambd  
    return signal
    
def make_filter(signal, offset_x):
    """Obtain filter from response to step function
    
    Takes derivative of Heaviside to get Dirac. Avoid zeros at both ends.
    """
    # impulse response. Step function is integration of dirac delta
    hvsd = signal[(offset_x):]
    dirac = np.gradient(hvsd)# + offset_y
    dirac = dirac[dirac > 0.0001]
    return dirac, hvsd

def get_step(time, offset_x, offset_y, reps = 4):
    """"Creates true step response
    """
    ramp_up = np.heaviside(time, 0)
    ramp_down = 1 - ramp_up
    step =  np.hstack(( np.zeros(offset_x),
                        np.tile(np.hstack((ramp_up, ramp_down)), reps),
                        np.zeros(offset_x))) + offset_y          
    return step

def get_exp_data(fpath, sampling_time):
    data = pd.read_csv(fpath, sep = ",", skiprows = 8).iloc[:, 1].values
    data = resample_poly(data, sampling_time, 1)
    return data
    
def get_filter_from_data(data, time, offset_x, offset_y, st):
    data = data[4500*st:5300*st]
    # normalize, downsample, trim, offset from 0
    data = ((data - min(data)) / (max(data) - min(data))) 
    factor = np.int(len(data) / (len(time) + 2*offset_x))
    data = decimate(data, factor)
    data = data[(len(data) - len(time) - 2*offset_x):]
    data = data + offset_y
    # get response to step and derivative = dirac delta
    hvsd = data
    dirac = np.gradient(hvsd)
    dirac = dirac[dirac > 0.0001]
    return dirac, hvsd
    
def apply_filter(data, filter):
    signal = wiener_deconvolution(data, filter)[:len(data)]
    return signal
    
# Worst case scenario from specs : signal Time t98%  < 60 s at 25 °C
multiplier = erfinv(0.98)/60
offset_y = .01
offset_x = 300
reps = 1
time = np.arange(301)
lambd = 0
sampling_time = 3 #s

fpath = ''.join(("P:\\MartinHolub\\phos_lifetime\\",
    "2018-01-18-pO2-calibrator-test006\\sensor_data\\",
    "Experiment_18012018_9ul2ml_air_recycled_2018-01-18 14-23-00.csv"))
data = get_exp_data(fpath, sampling_time)

# ## Alternative 1 - filter is the step   
# #signal = get_signal(time, offset_x, offset_y, 0.5)
# #step = get_step(time, offset_x, offset_y)

## Alternative 2 - filter is the observation
signal = get_step(time, offset_x, offset_y, reps = reps)
filter = get_signal(  time, offset_x, offset_y, reps = 0.5, lambd = lambd)
filter, hvsd = make_filter(filter, offset_x)
#filter, hvsd = get_filter_from_data(data, time, offset_x, offset_y, sampling_time) 
observation = get_signal(   time, offset_x, offset_y, reps = reps, lambd = lambd)
assert len(signal) == len(observation)
observation_est = convolve(signal, filter, mode = "full")[:len(observation)]

pdf = PdfPages('templates.pdf')
fig, ax = plt.subplots(2,2, frameon = False)
ax[0,0].plot(signal, color = 'k', label= "signal")
ax[0,1].plot(observation, color = 'r', label = "observation")
ax[0,1].plot(observation_est, color = 'b', label = "observation_est")
ax[1,0].plot(filter, color = 'g', label = "filter")
ax[1,1].plot(hvsd, color = 'b', label = "response to step")
for AX in ax.reshape(-1):
    AX.legend()
pdf.savefig()
plt.close()
pdf.close()

signal_est = wiener_deconvolution(observation, filter, lambd)[:len(observation)]
filt_est  = wiener_deconvolution(observation, signal, lambd)[:len(filter)]

signal_est1, _ = deconvolve(observation, filter)
filt_est1, _ = deconvolve(observation, signal)

signal_est2 = deconvolve_fun(observation, filter)[:len(observation)]
filt_est2 = deconvolve_fun(observation, signal)[:len(filter)]

pdf = PdfPages('results.pdf')
fig, ax = plt.subplots(3,2, frameon = False)
ax[0,0].plot(signal_est, color = 'b', label= "signal_est(wiener)")
ax[0,0].plot(signal, color = 'r', alpha = 0.3, label= "true signal")
ax[0,1].plot(filt_est, color = 'g', label = "filt_est(wiener)")
ax[0,1].plot(filter, color = 'r', alpha = 0.3, label = "true filter")
ax[1,0].plot(signal_est1, color = 'b', label = "signal_est(numpy)")
ax[1,0].plot(signal, color = 'r', alpha = 0.3, label= "true signal")
ax[1,1].plot(filt_est1, color = 'g', label = "filt_est(numpy)")
ax[1,1].plot(filter, color = 'r', alpha = 0.3, label = "true filter")
ax[2,0].plot(signal_est2, color = 'b', label = "signal_est(fun)")
ax[2,0].plot(signal, color = 'r', alpha = 0.3, label= "true signal")
ax[2,1].plot(filt_est2, color = 'g', label = "filt_est(fun)")
ax[2,1].plot(filter, color = 'r', alpha = 0.3, label = "true filter")
for AX in ax.reshape(-1):
    AX.legend()
pdf.savefig()
plt.close()
pdf.close()

data_est = apply_filter(data, filter)
st = sampling_time
pdf = PdfPages('real_data_test.pdf')
fig, ax = plt.subplots(2,1, frameon = False)
ax[0].plot(data_est[200:-200], color = 'b', label= "data_est(wiener)")
ax[0].plot(data[200:-200], color = 'r', label = "data")
ax[1].plot(data_est[4500*st:5300*st], color = 'b', label= "data_est(wiener)")
ax[1].plot(data[4500*st:5300*st], color = 'r', label = "data")
for AX in ax.reshape(-1):
    AX.legend()
pdf.savefig()
plt.close()
pdf.close()
