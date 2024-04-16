# pylint: disable=invalid-name
"""
Module for L1 processing following the description in Tello (2010). The description in the patent is partly vague. We
use the gradient of the ringdown to determine the impedance through the optimization algorithm.

Tello, L.N., 2010. Ultrasonic logging methods and apparatus for automatically calibrating measures of acoustic
impedance of cement and other materials behind casing. US Patent No. US 7,755,973 B2
"""

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import hilbert
from scipy.optimize import minimize
from numba import guvectorize, float64, int64, complex128, prange, njit

from pyintegrity import Quantity
from ..series import PulseEchoSeries
from ...logchannel import LogChannel
from .result import ProcessingResult
from .helpers import find_peak

#%%
class L1result(ProcessingResult):
    """Result container for L1 processing
    
    Args:
        L1: gradient of ringdown tail
        fres: resonance frequency derived from group-delay minimum
        Spec_firstpulse: the spectrum from the first pulse used in the convolution with the 1D model
    """
    def __init__(self, L1: LogChannel,
                    fres: LogChannel,
                    Spec_firstpulse: LogChannel) -> None: 
        super().__init__()                         
        
        self.L1= L1
        self.fres= fres
        self.Spec_firstpulse= Spec_firstpulse
        self.thickness: LogChannel
        
    def calculate_thickness(self,series: PulseEchoSeries) -> None:
        """Calculate thickness from resonance frequency
        
        Args: 
            series: `PulseEchoSeries` object containing the waveform data
        """
        assert series.casing is not None and series.casing.material is not None \
            and series.casing.material.speed is not None
        d=Quantity((series.casing.material.speed.magnitude/(2*self.fres.data.to_numpy())),'m')
        self.thickness= LogChannel(d, z=series.z, phi=series.phi)
        
    def derive_impedance(self,series: PulseEchoSeries,
                         Trt: np.ndarray,
                         freq: Quantity,
                         win_lim: list) -> None:
        """Derive impedance using the L1 gradient value through optimization algorithm 
        
        Args: 
            series: `PulseEchoSeries` object containing the waveform data
            Trt: transfer function, spectrum of initial pulse
            freq: frequency vector
            win_lim: window limits for calculation of ringdown gradient
        """
        dt = 1/series.sampling_freq
                
        L1data = self.L1.data.to_numpy()
        d = self.thickness.data.to_numpy()
        [M,N] = L1data.shape

        assert series.casing is not None and series.casing.material is not None and series.inner_material is not None
        casing_material = series.casing.material
        inner_material = series.inner_material
        assert casing_material.speed is not None and casing_material.impedance is not None
        assert inner_material.impedance is not None
        
        Z_res=np.zeros([M,N])
        initial_val =5*10**6
        for i in prange(M):   # pylint: disable=not-an-iterable
            for j in prange(N):   # pylint: disable=not-an-iterable
                if ~np.isnan(d[i,j]):
                    argsH=(freq.magnitude,
                           casing_material.speed.magnitude,
                           d[i,j],
                           casing_material.impedance.magnitude,
                           inner_material.impedance.magnitude[i],
                           dt.magnitude,
                           Trt[i,j,:],
                           L1data[i,j],
                           win_lim[0][i,j],
                           win_lim[1][i,j])
                    try:
                        final_L2 = minimize(L1_optimize, initial_val, args=argsH,method='Nelder-Mead',tol=1000)
                        Z_res[i,j]=final_L2.x*10**-6
                    except:   # FIXME: Specify exception type   # pylint: disable=bare-except
                        Z_res[i,j]=np.nan
                else:
                    Z_res[i,j]=np.nan
                         
        self.impedance = LogChannel(Quantity(Z_res,'MRayl'), z=series.z, phi=series.phi)

def process_L1(series: PulseEchoSeries,
               interval_f: tuple[float, float] | None = None,
               limits_windows: tuple[float, float, float] | None = None,
               fft_scale: int | None = None)-> L1result:
    """Derive pipe thickness and impedance based on the L1 algorithm
    
    Args:
        series: `PulseEchoSeries` containing input waveform data
        interval_f: limits for frequency search window from expected resonance frequency in Hz, default (-40000,40000)
        limits_windows: define window size as number of periods from maximum amplitude for the window defining the \
            initial pulse (symmetric) and the window to derive the L1 parameter, default (1.5,3,9)
        fft_scale: scaling factor for higher frequency resolution
        
    Returns:
        L1Result: object containing impedance and thickness
    """   
    # Frequency interval
    if interval_f is None:
        f_search=(-40000.,40000.)
    elif isinstance(interval_f, (list, tuple)):
        if len(interval_f)!=2:
            print('The search interval needs to be given as start and stop frequency deviation from the expected '
                  'resonance frequency in Hz. The default range [-40000, 40000] will be used.')
            f_search=(-40000.,40000.)
        else:
            f_search=interval_f 
    else:
        print('The search interval needs to be a tuple or a list with two values. The default range [-40000, 40000] '
              'will be used.')
        f_search=(-40000,40000)   

     # Window length normalization window, processing window   
    if limits_windows is None:
        lim_win=(1.5, 3., 9.)
    elif isinstance(limits_windows, (list, tuple)):
        if len(limits_windows)!=3:
            print('Start and stop limits for the windows, 3 values as number of periods from the max pulse amplitude '
                  'should be given. Symmetric value for the window defining the first pulse, and start and stop of the '
                  'window for the L1 calculation. Default used (1.5,3,9)')
            lim_win=(1.5, 3., 9.)
        else:
            lim_win=limits_windows 
    else:
        print('Start and stop limits for the windows, 3 values as number of periods from the max pulse amplitude '
              'should be given. Symmetric value for the window defining the first pulse, and start and stop of the '
              'window for the L1 calculation. Default used (1.5,3,9)')
        lim_win=(1.5, 3., 9.)    
     
     # Scaling for fft resolution
    if fft_scale is None:
        scale=10
    else:
        scale=fft_scale
        
    TrO, Trt, freq, env, win_lim = L1_preprocessing(series,lim_win,scale)    
    Spec_firstpulse = LogChannel(Trt, z=series.z, phi=series.phi , f=freq)

    assert series.casing is not None and series.casing.material is not None \
        and series.casing.thickness_nominal is not None
    f_exp = series.casing.material.speed/(2*series.casing.thickness_nominal)
    f_resonance=find_resonance(TrO,freq,f_exp,f_search)
    fres = LogChannel(f_resonance, z=series.z, phi=series.phi)

    dt = 1/series.sampling_freq
    L1data=_L1_gradient(env,dt.magnitude,win_lim[0],win_lim[1])   #pylint: disable=typecheck
    L1 = LogChannel(L1data, z=series.z, phi=series.phi)
       
    result=L1result(L1,fres,Spec_firstpulse)
    
    # Calculate thickness
    if fres is None:    # No resonance frequency detected
        return result   # calculation cannot proceed further
    result.calculate_thickness(series)
    
    if L1 is None:      # No L1 gradient derived
        return result   # calculation cannot proceed further
    result.derive_impedance(series,Trt,freq,win_lim)

    return result


#%%
@guvectorize([(int64, int64, int64, float64[:],complex128[:])], '(),(),(),(n)->(n)', nopython=True, target='parallel')           
def _tukey_Spec(indTS: int, indTE:int,  Nfft: int, waveform: np.ndarray, Tr: np.ndarray):
    """Calculate spectrum of first pulse using a Tukey window  
    https://leohart.wordpress.com/2006/01/29/hello-world/
    
    Args:
        indTS: start normalization window
        indTE: stop normalization window
        Nfft: Upscaled Nfft length
        waveform: waveform tapered with zeros for higher resolution frequency result
        
    Returns:
        Tr: transfer function, spectrum from initial pulse 
    """
    # 
    alpha=0.5
    window_length = (indTE- indTS)*2   
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
    Tr[:]=fft(waveform[indTS:indTE]*w[(indTE- indTS)::],Nfft)
#%%
def L1_preprocessing(series: PulseEchoSeries,
                     wlim: tuple[float, float, float],
                     scale: int) -> tuple[np.ndarray, np.ndarray, Quantity, np.ndarray, list]:
    """ Preprocessing of the data to derive group delay curve, resonance frequency, group delay minimum and bandwidth \
        of minimum
    
    Args:
        series: `PulseEchoSeries` containing input waveform data
        wlim: limits for windows as periods from max amplitude, first value symmetric to derive first pulse, 2. and \
            3. value start and stop to derive L1 gradient
        scale: scaling factor for higher resolution for frequency resolution
    
    Returns:
        TrO: transfer function, spectrum of entire pulse
        Trt: transfer function, spectrum of initial pulse
        freqW: frequency vector
        env: envelope of signals
        [indT2,indT3]: window limits for calculation of ringdown gradient
    """
    assert series.casing is not None and series.casing.material is not None \
        and series.casing.material.speed is not None and series.casing.thickness_nominal is not None

    dt = 1/series.sampling_freq
    f_exp = series.casing.material.speed/(2*series.casing.thickness_nominal)
    waveforms = series.data.to_numpy()
    t: np.ndarray = series.data.t.to_numpy()
    env: np.ndarray = np.abs(hilbert(waveforms))
    env= 20*np.log10(env)
    t_max,_ = find_peak(t,env)
    [M,N,Nfft]=waveforms.shape
    
    # Find sample number for window
    t_W=[t_max-1/f_exp.magnitude*wlim[0],
         t_max+1/f_exp.magnitude*wlim[0],
         t_max+1/f_exp.magnitude*wlim[1],
         t_max+1/f_exp.magnitude*wlim[2]]
    t = t[np.newaxis,np.newaxis,:]
    indT0 = np.argmin(np.abs(t - t_W[0][:,:,np.newaxis]),axis=-1)   # First sample index of the range
    indT1 = np.argmin(np.abs(t - t_W[1][:,:,np.newaxis]),axis=-1)   # second sample index of the range
    indT2 = np.argmin(np.abs(t - t_W[2][:,:,np.newaxis]),axis=-1)   # third sample index of the range
    indT3 = np.argmin(np.abs(t - t_W[3][:,:,np.newaxis]),axis=-1)   # third sample index of the range
    
    # Windowed data
    NfftS=Nfft*scale
    waveformsScale=np.zeros([M,N,NfftS])
    waveformsScale[:,:,0:Nfft]=waveforms
    
    Trt=_tukey_Spec(indT0,indT1,NfftS,waveformsScale)  # initial pulse for 1D plane wave model #pylint: disable=typecheck
    TrO=_tukey_Spec(indT0,indT3,NfftS,waveformsScale)  # to derive thickness    #pylint: disable=typecheck
    
    freqW=np.arange(0,NfftS)/NfftS/((dt))
    
    return TrO, Trt, freqW, env, [indT2,indT3]
#%%
def find_resonance(TrO: np.ndarray, f: Quantity, f_exp: Quantity, fW: tuple[float, float]) -> Quantity:
    """ Derive resonance frequency from spectrum from entire signal
    
    Args:
        TrO: transfer function, spectrum of entire pulse
        f: frequency vector
        f_exp: expected resonance frequency from nominal thickness
        fW: Search window size plus/minus
   
    Returns:
        fres: resonance frequency
    """
    indf0 = np.argmin(np.abs(f.magnitude - (f_exp.magnitude+fW[0])))  # First sample index of the range
    indf1 = np.argmin(np.abs(f.magnitude - (f_exp.magnitude+fW[1])))
    
    fres,_=find_peak(f[indf0:indf1],-20*np.real(np.log10(TrO[:,:,indf0:indf1])))
    assert isinstance(fres, Quantity)
    return fres

#%% 
@guvectorize([(float64[:], float64, int64, int64, float64[:])], '(n),(),(),()->()', nopython=True, target='parallel')      
def _L1_gradient(env,dt,indTS,indTE,L1):
    """ linear regression to derive L1 parameter, gradient over the ringdown window
    
    Args:
        env: envelope of signal 
        dt: time increment
        indTS: start window
        indTE: stop window
        
    Returns:
        L1: gradient of ringdown
    """
    x=np.arange(0,(indTE-indTS))*dt
    y=env[indTS:indTE]
    w=indTE-indTS
    sx = sum(x) 
    sy = sum(y) 
    sx2 = sum(x**2) 
    sxy = sum(x * y) 
    L1[0] = (w * sxy - sx * sy) / (w * sx2 - sx**2)
#%%
@njit
def hilbert1(u: np.ndarray) -> np.ndarray:
    """ implementation of Hilbert transform for usage with numba
    # https://stackoverflow.com/questions/56380536/hilbert-transform-in-python
    
    Args:
        u: input signal
   
    Returns:
        v: derived Hilbert transform
    """
    N = len(u)
    # take forward Fourier transform
    U: np.ndarray = fft(u)
    M = N - N//2 - 1
    # zero out negative frequency components
    U[N//2+1:] = np.array([0] * M)   # pylint: disable=unsupported-assignment-operation
    # double fft energy except @ DC0
    U[1:N//2] = np.array(2 * U[1:N//2])   # pylint: disable=unsupported-assignment-operation
    # take inverse Fourier transform
    v = ifft(U)
    return v

#%%
@njit
def _L1_gradient1(env: np.ndarray, dt: float, indTS: int, indTE: int) -> float:
    """ linear regression to derive L1 parameter, gradient over the ringdown 
    window, for just one trace
    
    Args:
        env: envelope of signal 
        dt: time increment
        indTS: start window
        indTE: stop window
    
    Returns:
        L1: gradient of ringdown
    """
    x=np.arange(0,(indTE-indTS))*dt
    y=env[indTS:indTE]
    w=indTE-indTS
    sx = sum(x) 
    sy = sum(y) 
    sx2 = sum(x**2) 
    sxy = sum(x * y) 
    return (w * sxy - sx * sy) / (w * sx2 - sx**2)    

@njit
def L1_planewave(freqW: np.ndarray, vpCAS: float, Tcas: float, Z_cas: float, Zin: float, Zout: float) -> np.ndarray:
    """ Plane wave calculation using derived thickness and assumption for
    outer-material impedance
    
    Args:
        freqW: frequency vector 
        vpCAS: P-wave speed of casing
        Tcas: derived thickness of casing
        Z_cas: impedance of casing
        Zin: impedance of inner fluid
        Zout: assumption of outer-material impedance
    
    Returns:
        Tr: reflection coefficient form plane wave model
    """
    beta = 2* np.pi*freqW/vpCAS
    A = np.cos(beta*Tcas)  
    B = -Z_cas*1j*np.sin(beta*Tcas)
    C= -1/Z_cas*1j*np.sin(beta*Tcas)
    D = np.cos(beta*Tcas) 
    Tr = (A*Zin-D*Zout-C*Zin*Zout+B)/(C*Zin*Zout-D*Zout-A*Zin+B) 
    return Tr

def L1_model(freqW: np.ndarray, vpCAS: float, Tcas: float, Z_cas: float, Zin: float, Zout: float, Trt: np.ndarray
             ) -> np.ndarray: 
    """ Calculation of modelled envelope based on the plane wave model
    
    Args:
        freqW: frequency vector 
        vpCAS: P-wave speed of casing
        Tcas: derived thickness of casing
        Z_cas: impedance of casing
        Zin: impedance of inner fluid
        Zout: assumption of outer-material impedance
    
    Returns:
        env: envelope derived  from modelled reflection coefficient
    """
    Tr = L1_planewave(freqW,vpCAS,Tcas,Z_cas,Zin,Zout)
    Nfft=len(Tr)
    data_conv=Tr*Trt
    data_mod=np.real(ifft(data_conv[:int(Nfft/2)]*2,Nfft))
    env = np.abs(hilbert(data_mod))
    env= 20*np.log10(env)
    return env

def L1_optimize(in_val: float, freqW: np.ndarray, vpCAS: float, Tcas: float, Z_cas: float, Zin: float, dt: float,
                Trt: np.ndarray, L1data: float, indTS: int, indTE: int) -> float:
    """Algorithm that calls model to optimize error between real data and model data to find the best impedance estimate
    
    Args:
        in_val: input value of impedance in MRayl
        freqW: frequency vector 
        vpCAS: P-wave speed of casing
        Tcas: derived thickness of casing
        Z_cas: impedance of casing
        Zin: impedance of inner fluid
        dt: time increment
        Trt: transfer function, spectrum of initial reflection
        L1data: reverberation  gradient from data 
        indTS: start for window to derive L1 gradient 
        indTE: stop for window to derive L1 gradient 
    
    Returns:
        error between gradient of measured and modelled data
    """
    in_val=in_val*10**-6

    data_modW = L1_model(freqW,vpCAS,Tcas,Z_cas,Zin,in_val,Trt)
    L1mod=_L1_gradient1(data_modW,dt,indTS,indTE)
    return (L1mod-L1data)**2
