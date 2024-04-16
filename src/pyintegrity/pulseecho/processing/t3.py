# pylint: disable=invalid-name
"""
Module for T3 processing following the description in Hayman (1991) and
Wright (1993).
The algorithm is based on the derivation of the group delay curve to derive 
thickness and impedance

Hayman, A., Hutin, R., Wright, P., 1991. High-resolution cementation
and corrosion imaging by ultrasound, in: SPWLA 32th Annual Logging
Symposium, p. 25.

Wright, P., 1993. Method and apparatus for the acoustic investigation of a
casing cemented in a borehole. US Patent No. 5,216,638.
"""

import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import hilbert
from scipy.optimize import minimize
from numba import guvectorize, float64, int64, complex128, prange, njit

from pyintegrity import Quantity
from ..series import PulseEchoSeries
from ...logchannel import LogChannel
from .result import ProcessingResult
from .helpers import find_peak

#%%
class T3result(ProcessingResult):
    """Result container for T3 processing
    
    Args:
        band: bandwidth of group delay minimum
        fres: resonance frequency derived from group-delay minimum
        Spec_firstpulse: the spectrum from the first pulse used in the convolution with the 1D model
    """
    def __init__(self, band: LogChannel,
                    fres: LogChannel,
                    Spec_firstpulse: LogChannel)-> None: 
        super().__init__()                          # Initializes impedance behind casing and casing thickness
        
        self.band= band
        self.fres= fres
        self.Spec_firstpulse= Spec_firstpulse
        
    def derive_impedance_thickness(self,series: PulseEchoSeries,
                          T3band_meas: Quantity,
                          T3freq: Quantity,
                          f_search: tuple[float, float],
                          indT_S0: np.ndarray,
                          indT_S1: np.ndarray,
                          indT_N0: np.ndarray,
                          indT_N1: np.ndarray) -> None:
        """Derive impedance using group delay parameters through optimization algorithm 
        
        Args: 
            series: `PulseEchoSeries` object containing the waveform data
            T3band_meas: bandwidth derived form measured data   
            T3freq: frequency vector
            f_search: window limits for search area
            indT_S0: signal window start
            indT_S1: signal window end
            indT_N0: normalization window start
            indT_N1: normalisation window end
        """
        assert series.casing is not None and series.casing.material is not None \
            and series.casing.material.speed is not None and series.casing.material.impedance is not None
        assert series.casing.thickness_nominal is not None
        assert series.inner_material is not None and series.inner_material.impedance is not None

        fres_meas=self.fres.data.to_numpy()
        T3N=self.Spec_firstpulse.to_numpy()
        f_exp= series.casing.material.speed/(2*series.casing.thickness_nominal)
        # Optimization to derive outer-material impedance and pipe thickness
        d= series.casing.material.speed.magnitude/2/(fres_meas)  # derive thickness from group delay resonance

        [M,N]=T3band_meas.shape
        Z_res=np.zeros([M,N])
        d_res=np.zeros([M,N])

        for i in prange(M):   # pylint: disable=not-an-iterable
            for j in prange(N):   # pylint: disable=not-an-iterable
                if ~np.isnan(d[i,j]):
                    initial_val =[5*10**6, d[i,j]]
            
                    indT_S=[indT_S0[i,j],indT_S1[i,j]]
                    indT_N=[indT_N0[i,j],indT_N1[i,j]]
                    try:
                        argsH=(series.casing.material.impedance.magnitude,
                               series.inner_material.impedance.magnitude[i],
                               series.casing.material.speed.magnitude,
                               indT_S, indT_N,T3N[i,j,:],
                               f_exp.magnitude,
                               T3freq.magnitude,
                               fres_meas[i,j],
                               f_search,
                               T3band_meas.magnitude[i,j])
                        final_T3 = minimize(T3_optimize, initial_val, args=argsH,method='Nelder-Mead',tol=1000)
                        Z_res[i,j]=final_T3.x[0]*10**-6
                        d_res[i,j]=final_T3.x[1]
                        
                    except:   # FIXME: Specify exception type   # pylint: disable=bare-except
                        Z_res[i,j]=np.nan
                        d_res[i,j]=np.nan
                else:
                    Z_res[i,j]=np.nan
                    d_res[i,j]=np.nan
        self.impedance = LogChannel(Quantity(Z_res,'MRayl'), z=series.z, phi=series.phi)
        self.thickness = LogChannel(Quantity(d_res,'m'), z=series.z, phi=series.phi)
        
    def calculate_thickness_fres(self,series: PulseEchoSeries) -> None:
        """Calculate thickness from resonance frequency
        
        Args: 
            series: `PulseEchoSeries` object containing the waveform data
        """
        assert series.casing is not None and series.casing.material is not None \
            and series.casing.material.speed is not None

        d=Quantity((series.casing.material.speed.magnitude/(2*self.fres.data.to_numpy())),'m')
        self.thickness= LogChannel(d, z=series.z, phi=series.phi)

def process_T3(series: PulseEchoSeries,
               interval_f: tuple[float, float] | None = None,
               limits_windows: tuple[float, float] | None = None,
               fft_scale: int | None = None)-> T3result:
    """Derive pipe thickness and impedance based on the T3 algorithm
    
    Args:
        series: `PulseEchoSeries` containing input waveform data
        interval_f: limits for frequency search window from expected resonance frequency in Hz, default (-60000.,60000.)
        limits_windows: give the window size in periods for the normalisation window (symmetric) and the end of the \
            signal window from max amplitude. Default= (2.5,7)
        fft_scale: value to scale fft resolution, default value = 10
        
    Returns:
        T3result: object containing impedance and thickness
    """
    assert series.casing is not None and series.casing.material is not None and series.casing.material.speed is not None
    assert series.casing.thickness_nominal is not None

    # Frequency interval
    if interval_f is None:
        f_search=(-60000.,60000.)
    elif isinstance(interval_f, (list, tuple)):
        if len(interval_f)!=2:
            print('The search interval needs to be given as start and stop frequency deviation from the expected ' 
                  'resonance frequency in Hz. The default range [-60000, 60000] will be used. ')
            f_search=(-60000.,60000.)
        else:
            f_search=interval_f 
    else:
        print('The search interval needs to be a tuple or a list with two values. The default range [-60000, 60000] '
              'will be used. ')
        f_search=(-60000.,60000.)
        
    # Window length normalization window, processing window   
    if limits_windows is None:
        lim_win=(2.5,7.)
    elif isinstance(limits_windows, (list, tuple)):
        if len(limits_windows)!=2:
            print('Window length for normalization and signal window, two values should be given. The default window '
                  'length of 2.5 and 7 periods will be used.')
            lim_win=(2.5,7.)
        else:
            lim_win=limits_windows 
    else:
        print('Window length for normalization and signal window, two values should be given. The default window '
              'length of 2.5 and 7 periods will be used.')
        lim_win=(2.5,7.)    
    
    # Scaling for fft resolution
    if fft_scale is None:
        scale=10
    else:
        scale=fft_scale
    
    f_exp= series.casing.material.speed/(2*series.casing.thickness_nominal)
    
    # Preprocessing of data
    T3freq,T3taugSN_meas, T3N, indT_S0, indT_S1, indT_N0, indT_N1=PreStep_processing(series,lim_win,scale)

    # Group delay minimum, group delay bandwidth and resonance frequency from group-delay curve
    fres_meas,taumin_meas=T3_tauF_tauMin(T3freq, f_exp, f_search, T3taugSN_meas)
    T3band_meas=T3_band(T3freq, fres_meas,taumin_meas, T3taugSN_meas)
    
                
    Spec_firstpulse = LogChannel(T3N, z=series.z, phi=series.phi , f=T3freq)          
    fres = LogChannel(fres_meas, z=series.z, phi=series.phi)
    band = LogChannel(T3band_meas, z=series.z, phi=series.phi)
    result = T3result(band,fres,Spec_firstpulse)
    
    # derive thickness and impedance
    if fres is None:    # No resonance frequency detected
        return result   # calculation cannot proceed further
    if band is None:    # no bandwidth detected
        print('Thickness is calculated from resonance frequency of group delay minimum. '
              'No optimization algorithm used.')
        result.calculate_thickness_fres(series)
        return result
    result.derive_impedance_thickness(series, T3band_meas, T3freq, f_search, indT_S0, indT_S1, indT_N0, indT_N1)

    return result
#%%
@guvectorize([(int64, int64, int64, float64[:],complex128[:])], '(),(),(),(n)->(n)', nopython=True, target='parallel')           
def _nom_Spec(indT_N0: int, indT_N1:int, Nfft: int, waveforms: np.ndarray, T3N_w: np.ndarray):
    """calculate normalization spectrum for correct window  
    
    Args:
        indT_N0: start normalization window
        indT_N1: stop normalization window
        Nfft: Upscaled Nfft length
        waveforms: waveforms tapered with zeros for higher resolution frequency result
        
    Returns:
        T3N_w: normalization spectrum for given window
    """
    N_ham=np.hamming(indT_N1 - indT_N0)
    T3N_wa=np.zeros(Nfft,dtype=np.complex128)
    if indT_N0<0:
        T3N_wa[:int(Nfft/2+1)]=rfft(waveforms[0:indT_N1]*N_ham[np.abs(indT_N0):],Nfft)
    else:
        T3N_wa[:int(Nfft/2+1)]=rfft(waveforms[indT_N0:indT_N1]*N_ham,Nfft)
    T3N_w[:]=T3N_wa

@guvectorize([(int64, int64, int64, int64, float64[:],complex128[:])], '(),(),(),(),(n)->(n)',
             nopython=True, target='parallel')           
def _sig_Spec(indT_S0: int, indT_S1:int, indT_N0: int, Nfft: int, waveforms: np.ndarray, T3S_w: np.ndarray):
    """calculate signal spectrum for correct window  
    
    Args:
        indT_S0: start signal window
        indT_S1: stop signal window
        indT_N0: start normalization window
        Nfft: Upscaled Nfft length
        waveforms: waveforms tapered with zeros for higher resolution frequency result
        
    Returns:
        T3N_w: normalization spectrum for given window
    """
    S_ham=np.hamming(indT_S1- indT_S0)
    T3S_wa=np.zeros(Nfft,dtype=np.complex128)
    if indT_N0<0:
        T3S_wa[:int(Nfft/2+1)]=rfft(waveforms[0:indT_S1]*S_ham[np.abs(indT_S0):],Nfft)
    elif indT_N0>=0 and indT_S0 <0:
        T3S_wa[:int(Nfft/2+1)]=rfft(waveforms[indT_N0:indT_S1]*S_ham[np.abs(indT_S0)+indT_N0:],Nfft)
    else:
        T3S_wa[:int(Nfft/2+1)]=rfft(waveforms[indT_N0:indT_S1]*S_ham[np.abs(indT_S0-indT_N0):],Nfft)
    T3S_w[:]=T3S_wa
#%%       
def PreStep_processing(series: PulseEchoSeries, lim_win:tuple[float,float], scale: int
                       ) -> tuple[Quantity, Quantity, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Preprocessing of the data to derive group delay curve, resonance frequency, group delay minimum and \
        bandwidth of minimum
    
    Args:
        series: `PulseEchoSeries` containing input waveform data
        lim_win: window limits in periods for normalization and signal window
        scale: scale of frequency resolution
        
    Returns:
        T3freq: Frequency vector corresponding to the spectrum
        T3taugSN: group delay curve
        T3N: normalization spectrum, spectrum of the first pulse
        indT_S0: start signal window
        indT_S1: stop signal window
        indT_N0: start normalization window
        indT_N1: stop normalization window
    """
    assert series.casing is not None and series.casing.material is not None and series.casing.material.speed is not None
    assert series.casing.thickness_nominal is not None

    N_per=lim_win[0]
    S_per=lim_win[1]

    ### Step1 ###
    dt = 1/series.sampling_freq
    f_exp = series.casing.material.speed/(2*series.casing.thickness_nominal)
    waveforms = series.data.to_numpy()
    t = series.data.t.to_numpy()
    
    [M,N,Nfft]=waveforms.shape
    
        
    env = np.abs(hilbert(waveforms))
    env= 20*np.log10(env)
    T3t_max,_ = find_peak(t,env)
    t_help=np.append(np.arange(-dt.magnitude*len(t),0,dt.magnitude), t)[np.newaxis,np.newaxis,:]
    
    ### Step2,3 ###
    # find window length for normalization and signal window
    per = 1/f_exp.magnitude
    T3t_N = [T3t_max-(N_per*per), T3t_max+(N_per*per)]
    indT_N = [np.argmin(np.abs(t_help - T3t_N[0][:,:,np.newaxis]),axis=-1),   # First sample index of the range
              np.argmin(np.abs(t_help - T3t_N[1][:,:,np.newaxis]),axis=-1)]
    indT_N0=indT_N[0]-Nfft
    indT_N1=indT_N[1]-Nfft
    
    T3t_S = [T3t_max-(S_per*per), T3t_max+(S_per*per)]
    indT_S = [np.argmin(np.abs(t_help - T3t_S[0][:,:,np.newaxis]),axis=-1),   # First sample index of the range
              np.argmin(np.abs(t_help - T3t_S[1][:,:,np.newaxis]),axis=-1)]
    indT_S0=indT_S[0]-Nfft
    indT_S1=indT_S[1]-Nfft
    
    # Upscaling 
    NfftS=Nfft*scale
    waveformsScale=np.zeros([M,N,NfftS])
    waveformsScale[:,:,0:Nfft]=waveforms
    
    NfftS2=int(NfftS/2+1)
    T3N=_nom_Spec(indT_N0,indT_N1,NfftS,waveformsScale)[:,:,:NfftS2]   # pylint: disable=typecheck
    T3S=_sig_Spec(indT_S0,indT_S1,indT_N0,NfftS,waveformsScale)[:,:,:NfftS2]   # pylint: disable=typecheck
    
    T3freq=np.arange(0,NfftS2)/NfftS/((dt))
    df=T3freq[1]-T3freq[0]
    
    T3SN = T3S/T3N
    phirSN = np.unwrap(np.angle(T3SN))

    T3taugSN = (phirSN[:,:,1::]-phirSN[:,:,0:-1:])/(2*np.pi*df) 

    return T3freq,T3taugSN, T3N, indT_S0, indT_S1, indT_N0, indT_N1

#%%  
def T3_tauF_tauMin(f: Quantity, f0: Quantity, f_search: tuple[float, float], T3taugSN:Quantity
                   ) -> tuple[Quantity, Quantity]:
    """ Deriving resonance frequency and group delay minimum
    
    Args:
        f: Frequency vector corresponding to the spectrum
        f0: expected resonance frequency
        f_search: search window for group delay minimum as plus-minus from expected resonance frequency
        T3taugSN: group delay curve   
    
    Returns:
        tauF: resonance frequency from group delay minimum
        taumin: group delay minimum
    """
    # Get tau frequency and tau minimum
    f_fit = (f0.magnitude+f_search[0],f0.magnitude+f_search[1])
    fstart = np.argmin(np.abs(f.magnitude-f_fit[0]))
    fstop =  np.argmin(np.abs(f.magnitude-f_fit[1]))    
    i_range=np.arange(fstart,fstop)
    if T3taugSN.ndim==3:
        tauF,taumin=find_peak(f[i_range],T3taugSN[:,:,i_range])
    else:
        tauF,taumin=find_peak(f[i_range],T3taugSN[i_range])    
    return tauF,-taumin   # type: ignore[return-value]

#%%
@guvectorize([(float64[:], float64, int64, int64, float64, float64[:])], '(n),(),(),(),()->()',
             nopython=True, target='parallel')           
def _get_band(T3taugSN: np.ndarray, taumin: float, fstart: int, fstop: int, df: float, band: np.ndarray):
    """ Calculating the bandwidth and the group delay minimum
    
    Args:
        T3taugSN: group delay curve
        taumin: group delay minimum
        fstart: sample number start of window
        fstop: sample number stop of window
        df: frequency step size
    
    Returns:
        band: width of group delay minimum
    """
    if np.isnan(taumin):
        band[0]=np.nan
    else:
        i_range=np.arange(fstart,fstop)
        peak40 = taumin /100*60 # 40 % from maximum
        taudiff=-T3taugSN[i_range]-peak40
        taucr = np.where(np.sign(taudiff[:-1]) != np.sign(taudiff[1:]))[0] 
        if len(taucr)<=1:
            band[0]=np.nan
        else:
            taucrU=np.zeros(2)
            taucrU[0]=taucr[0] + (np.abs(taudiff[taucr[0]])*100
                                  / (np.abs(taudiff[taucr[0]])+np.abs(taudiff[taucr[0]+1]))
                                  ) / 100
            taucrU[1]=taucr[1] + (np.abs(taudiff[taucr[1]])*100
                                  / (np.abs(taudiff[taucr[1]])+np.abs(taudiff[taucr[1]+1]))
                                  ) / 100
            band[0] = (taucrU[1]-taucrU[0])*df
  
#%%
def T3_band(f: Quantity,tauF: Quantity,taumin: Quantity,T3taugSN: Quantity) -> Quantity:
    """ Calculating the bandwidth og the group delay minimum
    
    Args:
        f: Frequency vector corresponding to the spectrum
        tauF: resonance frequency of group delay minimum
        taumin: group delay minimum
        T3taugSN: group delay curve
    
    Returns:
        band: bandwidth of group delay minimum
    """
    df=f[1]-f[0]
    f_fit = (tauF.magnitude-tauF.magnitude*1/3,tauF.magnitude+tauF.magnitude*1/3)
    fstart = np.argmin(np.abs(f.magnitude[np.newaxis, np.newaxis, :]-f_fit[0][:, :, np.newaxis]),axis=-1)
    fstop =  np.argmin(np.abs(f.magnitude[np.newaxis, np.newaxis, :]-f_fit[1][:, :, np.newaxis]),axis=-1)
            
    band=_get_band(T3taugSN.magnitude, taumin.magnitude, fstart, fstop, df.magnitude)   # pylint: disable=typecheck
    band=Quantity(band,'Hz')
    return band   

@njit
def T3_band1(f: np.ndarray ,tauF: float, taumin: float, T3taugSN: np.ndarray) -> float:
    """ Calculating the bandwidth and the group delay minimum for one trace
    
    Args:
        f: Frequency vector corresponding to the spectrum
        tauF: resonance frequency of group delay minimum
        taumin: group delay minimum
        T3taugSN: group delay curve
    
    Returns:
        band: bandwidth of group delay minimum
    """
    df=f[1]-f[0]
    f_fit = (tauF-tauF*1/3,tauF+tauF*1/3)
    
    fstart = np.argmin(np.abs(f-f_fit[0]))
    fstop =  np.argmin(np.abs(f-f_fit[1]))
        
    if np.isnan(taumin):
        band=np.nan
    else:
        i_range=np.arange(fstart,fstop)
        peak40 = taumin /100*60 # 40 % from maximum
        taudiff=-T3taugSN[i_range]-peak40
        taucr = np.where(np.sign(taudiff[:-1]) != np.sign(taudiff[1:]))[0] 
        if len(taucr)<=1:
            band=np.nan
        else:
            taucrU=np.zeros(2)
            taucrU[0]=taucr[0] + (np.abs(taudiff[taucr[0]])*100
                                  / (np.abs(taudiff[taucr[0]])+np.abs(taudiff[taucr[0]+1]))
                                  ) / 100
            taucrU[1]=taucr[1] + (np.abs(taudiff[taucr[1]])*100
                                  / (np.abs(taudiff[taucr[1]])+np.abs(taudiff[taucr[1]+1]))
                                  ) / 100
            band = (taucrU[1]-taucrU[0])*df 
    return band  


def T3_tauF_tauMin1(f: np.ndarray, f0: float, f_search: list, T3taugSN: np.ndarray) -> tuple[float, float]:
    """ Deriving resonance frequency and group delay minimum for one dataset
    
    Args:
        f: Frequency vector corresponding to the spectrum
        f0: expected resonance frequency
        f_search: search window for group delay minimum as plus-minus from expected resonance frequency
        T3taugSN: group delay curve   
    
    Returns:
        tauF: resonance frequency from group delay minimum
        taumin: group delay minimum
    """
    # Get tau frequency and tau minimum
    f_fit = (f0+f_search[0],f0+f_search[1])
    fstart = np.argmin(np.abs(f-f_fit[0]))
    fstop =  np.argmin(np.abs(f-f_fit[1]))    
    i_range=np.arange(fstart,fstop)
    tauF,taumin=find_peak(f[i_range],T3taugSN[i_range])    
    return tauF,-taumin   # type: ignore[return-value]
#%%
@njit
def T3_planewavemodel(f: np.ndarray, d_cas: float, v_cas: float, Zcas: float, Zin: float, Zfo: float) -> np.ndarray:  
    """ Reflection coefficient from 1D model
    
    Args:        
        f: Frequency vector corresponding to the spectrum
        d_cas: derived pipe thickness
        v_cas: P-wave speed of casing
        Zcas: impedance of casing
        Zin: impedance of inner material
        Zfo: outer material impedance guess
    
    Returns:
        ref_coeff: 1D modelled reflection coefficient
    """
    ckd = np.cos((2 * np.pi * f) / v_cas *d_cas)
    skd = np.sin((2 * np.pi * f) / v_cas *d_cas)
    ref_coeff = ( ((1 - Zin/Zfo) * ckd) + ((Zcas/Zfo - Zin/Zcas) * skd * 1j)) \
                / ( ((1 + Zin/Zfo) * ckd) + ((Zcas/Zfo + Zin/Zcas) * skd * 1j))
    return ref_coeff

@njit
def _step2(ref_coeff:np.ndarray, T3N_w:np.ndarray, Nfft2:int, S_ham:np.ndarray, indT_S0:int, indT_S1:int, indT_N0:int
           ) -> np.ndarray:
    """ Calculate step one and two in T3 processing using numba
    
    Args:
        f: Frequency vector corresponding to the spectrum
        T3N_w: spectrum of initial reflection
        Nfft: length of fft/2
        S_ham: hamming window
        indT_S0: start for signal window
        indT_S0: stop for signal window
        indT_N0: start for normalization window
        
    Returns:
        TSN: modelled signal window/normalization window
    """
    Nfft=(Nfft2-1)*2
    step1a = ref_coeff*T3N_w      
    step1=irfft(step1a,Nfft)
    # into time domain after multiplication with processing window
    if indT_N0<0: 
        step2 = rfft(step1[0:indT_S1]*S_ham[np.abs(indT_S0):],Nfft)
    else:
        step2 = rfft(step1[0:indT_S1-indT_N0]*S_ham[np.abs(indT_S0)+indT_N0:],Nfft)
    TSN=step2/T3N_w 
    return TSN

@njit
def _angle_unwrap1(signal_phase: np.ndarray) -> np.ndarray:
    """Calculate the group delay (using only functions that Numba supports)
    
    Args:
        T3N_w: modelled signal window/normalization window
        
    Returns:
        signal_phase: phase of signal
    """
    dd = signal_phase[1:] - signal_phase[:-1]
    ddmod = np.mod(dd + np.pi, 2*np.pi) - np.pi
    ddmod[np.logical_and(ddmod == -np.pi, dd > 0)] = np.pi
    ph_correct = ddmod - dd
    signal_phase[1:] += ph_correct.cumsum()
    return signal_phase

def T3_model_group_delay(f: np.ndarray, ref_coeff: np.ndarray, T3N_w: np.ndarray, indT_S: list, indT_N: list
                         ) -> np.ndarray:
    """ Deriving group delay from modelled plane wave reflection coefficient
    
    Args:        
        f: Frequency vector corresponding to the spectrum
        ref_coeff: 1D modelled reflection coefficient
        T3N_w: spectrum of initial reflection
        indT_S: list of start and stop for signal window
        indT_N: list of start and stop for normalization window
    
    Returns:
        T3taugSN_mod: modelled group delay
    """
    S_ham=np.hamming(indT_S[1]- indT_S[0])
    df=f[1]-f[0]
    Nfft=len(f)
    TSN=_step2(ref_coeff,T3N_w,Nfft,S_ham,indT_S[0],indT_S[1],indT_N[0]) 
    step3 = np.zeros(Nfft)
    step3 = _angle_unwrap1(np.angle(TSN))

    T3taugSN_mod = (step3[1::]-step3[0:-1:])/(2*np.pi*df)

    return T3taugSN_mod    
#%%
def T3_model(in_val: list, Zcas: float, Zin: float, v_cas: float, indT_S: list, indT_N: list, T3N_w: np.ndarray, 
             f0: float, f: np.ndarray, f_search: list) -> tuple[float, float]:
    """ Deriving modelled group delay curve and determining, resonance frequency, minimum value and bandwidth from the \
        modelled data
    
    Args:
        in_val: list of input parameters to derive impedance and thickness
        Zcas: impedance of casing
        Zin: impedance of inner material
        v_cas: P-wave speed of casing
        indT_S: list of start and stop for signal window
        indT_N: list of start and stop for normalization window
        T3N_w: spectrum of initial reflection
        f0: expected resonance frequency
        f: Frequency vector corresponding to the spectrum
        f_search: search window for group delay minimum as plus-minus from
    
    Returns:
        tauF_mod: frequency of group delay minimum from modelled data
        band_mod: bandwidth of group delay minimum from modelled data
    """
    Zfo=in_val[0]*10**-6
    d_cas=in_val[1]
    # plane wave model estimation of reflection coefficent
    ref_coeff = T3_planewavemodel(f,d_cas,v_cas,Zcas,Zin,Zfo)
    # calculation of group delay
    T3taugSN_mod = T3_model_group_delay(f,ref_coeff,T3N_w,indT_S, indT_N)
    # derive resonance frequency and group delay minimum
    tauF_mod,taumin_mod=T3_tauF_tauMin1(f, f0, f_search, T3taugSN_mod)

    # derive band from model data
    band_mod=T3_band1(f, tauF_mod,taumin_mod, T3taugSN_mod)
    return tauF_mod,band_mod


def T3_optimize(in_val: list, Zcas: float ,Zin: float, v_cas: float, indT_S: list, indT_N: list, T3N_w: np.ndarray, \
                f0: float, f: np.ndarray, T3f0_meas: float, f_search: list, T3band_meas: float) -> float:
    """ Algorithm that calls model to optimize error between real data and model data to find the best estimate for \
        impedance and thickness
    
    Args:
        in_val: list of input parameters to derive impedance and thickness
        Zcas: impedance of casing
        Zin: impedance of inner material
        v_cas: P-wave speed of casing
        indT_S: list of start and stop for signal window
        indT_N: list of start and stop for normalization window
        indT_S: list of start and stop for signal window
        f0: expected resonance frequency
        f: Frequency vector corresponding to the spectrum
        T3f0_meas: resonance frequency derived from measured data
        f_search: search window for group delay minimum as plus-minus from 
        T3band_meas: bandwidth of group-delay minimum of measured data
    
    Returns:
        error between resonance frequency and group delay minimum of measured and modelled data
    """
    T3f0_model,T3band_model=T3_model(in_val,Zcas,Zin,v_cas,indT_S, indT_N,T3N_w,f0,f,f_search)
    return ( ((T3f0_model-T3f0_meas)/T3f0_meas)**2 \
            + ((T3band_model/T3f0_model-T3band_meas/T3f0_meas)/(T3band_meas/T3f0_meas))**2 \
            )**2
