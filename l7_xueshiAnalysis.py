# In[]
import numpy as np
import os
import lecroy
import matplotlib.pyplot as plt
from scipy.fft import *
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
datasets = ['electronic', 'shotnoise', 'squeezing', 'antisqueezing']
channels = ['C1', 'C2', 'C3', 'C4']

def get_file_list(dataset, channel):
    folder = os.path.join("./xueshi", dataset, channel)
    fns = sorted(os.listdir(folder))
    fns = [os.path.join(folder, fn) for fn in fns]
    return fns
electronic=np.array([])
shotnoise=np.array([])
squeezing=np.array([])
antisqueezing=np.array([])
files = {ds:
           {ch:get_file_list(ds, ch) for ch in channels}
           for ds in datasets}

for ds in datasets:
    data = np.array([[
        lecroy.read(f, scale=False)[2] for f in files[ds][ch]]
                     for ch in channels])
    data = data.squeeze()
    globals()[ds] = data

meta, times, data = lecroy.read(files['shotnoise']['C1'][0], scale=False)
dt = meta['horiz_interval']
fs = 1/dt
N = len(data[0])
t = np.linspace(0, fs*N, N, endpoint=False)
f = rfftfreq(N, dt)
# meta

print(electronic.shape, shotnoise.shape, squeezing.shape, antisqueezing.shape)
not_filtered=True
# In[]
if not_filtered:
    test_signal=antisqueezing[0,1]
plt.plot(f[:N//2], rfft(test_signal)[:N//2].T)
#define filter at 38MHz
f0 = 28.70e6
b,a=signal.iirnotch(f0, 500, fs=fs)
filtered = signal.filtfilt(b, a, test_signal, axis=-1)
plt.plot(f[:N//2], rfft(filtered)[:N//2])
plt.xlim([25e6,3e7])
# In[]
electronic =signal.filtfilt(b, a, electronic, axis=-1)
shotnoise = signal.filtfilt(b, a, shotnoise, axis=-1)
squeezing = signal.filtfilt(b, a, squeezing, axis=-1)
antisqueezing = signal.filtfilt(b, a, antisqueezing, axis=-1)
not_filtered = False
# In[]
plt.semilogy(f[:N//2], np.abs(fft(np.mean(antisqueezing,axis=1),axis=-1))[:,:N//2].T)
plt.semilogy(f[:N//2], np.abs(fft(np.mean(squeezing,axis=1),axis=-1))[:,:N//2].T)

# In[]
sum_el=np.sum(electronic,axis=0)
sum_sn=np.sum(shotnoise,axis=0)
sum_sq=np.sum(squeezing,axis=0)
sum_asq=np.sum(antisqueezing,axis=0)

spectrum_electronic=np.abs(np.mean(rfft(sum_el,axis=-1),axis=0))[:50000]
spectrum_shotnoise=np.abs(np.mean(rfft(sum_sn,axis=-1),axis=0))[:50000]
spectrum_sum_sq=np.abs(np.mean(rfft(sum_sq,axis=-1),axis=0))[:50000]
spectrum_sum_asq=np.abs(np.mean(rfft(sum_asq,axis=-1),axis=0))[:50000]
f_plus = f[:50000]

del electronic, shotnoise, squeezing, antisqueezing, sum_sq, sum_asq
#In[]
plt.semilogy(f_plus, spectrum_sum_asq,label='antisqueezing')
plt.semilogy(f_plus, spectrum_shotnoise,label='shotnoise')
plt.semilogy(f_plus, spectrum_sum_sq,label='squeezing')
plt.semilogy(f_plus, spectrum_electronic,label='electronic')
plt.semilogy(f_plus,gaussian_filter1d(spectrum_shotnoise, 10))
plt.legend()
plt.xlim([0, 5e7])

#In[]
filt_spectrum_electronic=gaussian_filter1d(spectrum_electronic, 20)
filt_spectrum_shotnoise=gaussian_filter1d(spectrum_shotnoise, 20)
filt_spectrum_sum_sq=gaussian_filter1d(spectrum_sum_sq, 5)
filt_spectrum_sum_asq=gaussian_filter1d(spectrum_sum_asq, 5)

plt.semilogy(f_plus, filt_spectrum_sum_asq,label='antisqueezing')
plt.semilogy(f_plus, filt_spectrum_sum_sq,label='squeezing')
plt.semilogy(f_plus, filt_spectrum_shotnoise,label='shotnoise')
plt.semilogy(f_plus, filt_spectrum_electronic,label='electronic')
plt.legend()
# In[]

fn_sn=filt_spectrum_shotnoise-filt_spectrum_electronic
#gauss averae filter
fn_sq=(filt_spectrum_sum_sq-filt_spectrum_electronic)/fn_sn
fn_asq=(filt_spectrum_sum_asq-filt_spectrum_electronic)/fn_sn

sq_dB=10*np.log10(fn_sq)
asq_dB=10*np.log10(fn_asq)
sn_dB=10*np.log10(fn_sn)

n_points=1000
ff_sq_dB=np.mean(sq_dB.reshape(-1, 10000//n_points), axis=-1)[1+n_points//100:n_points]
ff_asq_dB=np.mean(asq_dB.reshape(-1, 10000//n_points), axis=-1)[1+n_points//100:n_points]
f_new=f_plus[::10000//n_points][1+n_points//100:n_points]
plt.plot(f_plus, asq_dB)
plt.plot(f_new, ff_asq_dB,".")
# plt.plot(f_plus, sn_dB)
plt.plot(f_plus, sq_dB)
plt.plot(f_new, ff_sq_dB,".")
plt.xlim([0, 5e7])
plt.ylim([-10,10])

#In[]
Vsq = lambda P, Pth, eta, frel: 1 - 4*eta*np.sqrt(P/Pth) / ((1+np.sqrt(P/Pth))**2 + frel**2)
Vasq = lambda P, Pth, eta, frel: 1 + 4*eta*np.sqrt(P/Pth) / ((1-np.sqrt(P/Pth))**2 + frel**2)
Vrot = lambda P, Pth, eta, frel, theta: np.cos(theta*np.pi/180)**2 * Vsq(P, Pth, eta, frel) + np.sin(theta*np.pi/180)**2 * Vasq(P, Pth, eta, frel)

def Vrotfluct(P, Pth, eta, frel, theta, sigma_x=0, sigma_p=None):
    if sigma_p is None:
        sigma_p = sigma_x
    
    Vx = .5 * Vsq(P, Pth, eta, frel) * (1 + np.cos(2 * theta * np.pi/180) * np.exp(-2 * (sigma_x * np.pi/180)**2))
    Vp = .5 * Vasq(P, Pth, eta, frel) * (1 - np.cos(2 * theta * np.pi/180) * np.exp(-2 * (sigma_p * np.pi/180)**2))

    return Vx + Vp

def lin2db(lin):
    return 10 * np.log10(lin)

def db2lin(db):
    return 10**(db/10)

def calc_noise_var(list_traces:np.ndarray):
    return np.var(list_traces,axis=1)

def res_model(params, f, sq_spec, asq_spec):
    P=1
    Pth, eta, bw, theta_sq, theta_asq, sigma = params
    model_sq = lin2db(Vrotfluct(P, Pth, eta, f/bw, theta_sq, sigma))
    model_asq = lin2db(Vrotfluct(P, Pth, eta, f/bw, theta_asq, sigma))
    residuals = np.concatenate([sq_spec-model_sq, asq_spec-model_asq])
    return residuals


res = least_squares(res_model, [0.8,.7,4e6, 0, 90, 0.1], 
                bounds=([.01,.1,1e6,-1,89,-1], [20,1,1e8,1,110,100]), 
                args=[f_new, ff_sq_dB, ff_asq_dB])
print(res)
Pth, eta, bw, theta_sq, theta_asq, sigma = res.x
plt.plot(f_new, ff_sq_dB,".",markersize=1)
plt.plot(f_new, ff_asq_dB,".",markersize=1)
plt.plot(f_new, lin2db(Vrotfluct(1, Pth, eta, f_new/bw, theta_sq, sigma)))
plt.plot(f_new, lin2db(Vrotfluct(1, Pth, eta, f_new/bw, theta_asq, sigma)))