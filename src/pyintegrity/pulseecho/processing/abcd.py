# pylint: disable=invalid-name
"""
Module for ABCD processing following the description in Mandal (2000)
The algorithm is based on the derivation of the parameter Sw, the sum of the 
absolute value over the ringdown period.

The 1D plane wave model as given by Mandal has a 180-degree phase shift 
compared to reference solutions. This does not make a difference in context of 
this method as only the absolute values of the modelled waveform are used in 
the end. However, to have a correct result we changed this.

In Mandal the ABCD function is derived by combining the spectrum of multiple
measurements, here we use the spectrum of each of the measurements

Mandal, B., Standley, T.E., 2000. Method to determine self-calibrated 
circumferential cased bond impedance. US Patent 6,041,861.
"""

import numpy as np
from scipy.fft import fft, ifft
from numba import guvectorize, float32, float64, int64, complex128

from pyintegrity import Quantity
from ..series import PulseEchoSeries
from ...logchannel import LogChannel
from .result import ProcessingResult
from .helpers import find_peak


class ABCDresult(ProcessingResult):
    """Result container for ABCD processing
    
    Args:
        fres: resonance frequency to derive thickness
        Sw: the parameter Sw as defined in Mandal et al. (2000) derived from the measurement data
        Spec_firstpulse: the spectrum from the first pulse used in the convolution with the 1D model
    """
    def __init__(self, Sw: LogChannel,
                 fres: LogChannel,
                 Spec_firstpulse: LogChannel) -> None: 
        super().__init__()                          # Initializes impedance behind casing and casing thickness
        
        self.Sw= Sw
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
    
    def calculate_impedance(self,series: PulseEchoSeries) -> None:
        """Calculate impedance 
        
        Args: series: `PulseEchoSeries` object containing the waveform data
        """
        Z=_ABCD_Z(series,self.Spec_firstpulse.data.to_numpy(),self.thickness.data.data,self.Sw.data.to_numpy())
        self.impedance = LogChannel(Z, z=series.z, phi=series.phi)


def process_abcd(series: PulseEchoSeries)-> ABCDresult:
    """Derive pipe thickness and impedance based on the abcd algorithm
    
    Args:
        series: `PulseEchoSeries` containing input waveform data
        
    Returns:
        ABCDResult object containing impedance and thickness
    """
    f_resonance, Sw_arr, Spec_firstpulse_arr, fvec = Mandal_processing(series)  
    Sw= LogChannel(Sw_arr, z=series.z, phi=series.phi)
    fres = LogChannel(f_resonance, z=series.z, phi=series.phi)
    Spec_firstpulse = LogChannel(Spec_firstpulse_arr, z=series.z, phi=series.phi , f=fvec)
    result = ABCDresult(Sw,fres,Spec_firstpulse)
    
    
    # Calculate thickness and impedance
    if fres is None:    # No resonance frequency detected
        return result   # calculation cannot proceed further
    result.calculate_thickness(series)
    result.calculate_impedance(series)
    
    return result

#%%    
@guvectorize([(float64[:], float64, float64, float64, float64[:])], '(n),(),(),()->()', \
             nopython=True, target='parallel')
def _find_reflection(waveform: np.ndarray,f_exp: float,t0: float, dt: float,x_peak):
    """Find time of the max amplitude of the initial pulse using points a, b, c 
    and a polynomial fit
    
    Args:
        waveform: waveform to be analysed
        t: time vector corresponding to the waveform vector
        f_exp: the expected resonance frequency based on known speed in pipe
        and nominal thickness
        dt: sampling time step
        
    Returns:
        x_peak: time of maximum of first reflection
    """
    b_idx=np.argmax(np.abs(waveform))
    a_idx=np.argmax(np.abs(waveform[0:(b_idx-np.int64(1/f_exp/4/dt))]))
    c_idx=np.argmax(np.abs(waveform[(b_idx+np.int64(1/f_exp/4/dt)):]))+b_idx+np.int64(1/f_exp/4/dt)
    t=np.arange(len(waveform))*dt+t0
    x=[t[a_idx],t[b_idx],t[c_idx]]
    y=[np.abs(waveform[a_idx]),np.abs(waveform[b_idx]),np.abs(waveform[c_idx])]
    # Can't use np.polyfit in Numba; using direct quadratic solution from https://stackoverflow.com/a/717791/3468067
    a = x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])
    b = x[2]**2 * (y[0] - y[1]) + x[1]**2 * (y[2] - y[0]) + x[0]**2 * (y[1] - y[2])
    x_peak[0] = - b / (2*a)  
    
@guvectorize([(float64[:], int64, int64, float32[:])], '(n),(),()->()', nopython=True, target='parallel')           
def _get_Sw(waveform: np.ndarray, tstart: int, tstop: int, Sw: np.ndarray):
    """ Calculating the parameter Sw
    We normalize the parameter Sw to avoid problems with variations in window length
    
    Args:
        waveform: waveform to be analysed
        tstart: sample no start of window
        tstop: sample no stop of window
    
    Returns:
        Sw: parameter of the sum of the absolute amplitudes for the given window
    """
    i_range=np.arange(tstart,tstop)
    Sw[0]=np.sum(np.abs(waveform[i_range]))/len(i_range)

@guvectorize([(int64, int64, int64, float32[:],float64[:])], '(),(),(),(n)->(n)', nopython=True, target='parallel')               
def _res_Spec(tstart: int, tstop: int, Nfft: int, waveform: np.ndarray, Ha_Rwin_dB: np.ndarray):
    """Calculate spectrum to derive resonance frequency  
    
    Args:
        tstart: start window
        tstop: stop window
        Nfft: Upscaled Nfft length
        waveform: waveform tapered with zeros for higher resolution frequency result
        
    Returns:
        Ha_Rwin_dB: spectrum from initial pulse 
    """
    Ha_Rwin =fft(waveform[tstart:tstop],Nfft) 
    Ha_Rwin_dB[:]=20*np.log10(np.abs(Ha_Rwin))   

    
@guvectorize([(int64, int64, int64, float64[:],complex128[:])], '(),(),(),(n)->(n)', nopython=True, target='parallel')           
def _tukey_Spec(indTS: int, indTE:int,  Nfft: int, waveform: np.ndarray, S_Ha: np.ndarray):
    """Calculate spectrum of first pulse using a Tukey window  
    https://leohart.wordpress.com/2006/01/29/hello-world/
    
    Args:
        indTS: start normalization window
        indTE: stop normalization window
        Nfft: Upscaled Nfft length
        waveform: waveform tapered with zeros for higher resolution frequency result
        
    Returns:
        S_Ha: spectrum from initial pulse 
    """
    alpha=0.5
    window_length = (indTE- indTS)*2   
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
    S_Ha[:]=fft(waveform[indTS:indTE]*w[(indTE- indTS)::],Nfft)

def Mandal_processing(series: PulseEchoSeries) -> tuple[Quantity, np.ndarray, np.ndarray, Quantity]:
    """ Processing of data following description in Mandal
    
    Args:
        series: `PulseEchoSeries` containing input waveform data
        
    Returns:
        f_ha: resonance frequency
        S_w_orig: Sw parameter derived from waveform data
        S_Ha: Spectrum of the first pulse used in the convolution with the plane wave model result
        Hafreq: Frequency vector corresponding to the spectrum 
    """
    assert series.casing is not None and series.casing.material is not None and series.casing.material.speed is not None
    assert series.casing.thickness_nominal is not None

    dt = 1/series.sampling_freq
    f_exp = series.casing.material.speed/(2*series.casing.thickness_nominal)
    waveforms = series.data.to_numpy()
    t = series.data.t.to_numpy()
    [M,N,Nfft]=waveforms.shape
    Nfft2=int(Nfft/2)
    Hafreq=np.arange(0,Nfft)/Nfft/((dt))
    
    # Fine time of maximum of first reflection
    T_peak = _find_reflection(waveforms,f_exp.magnitude,t[0],dt.magnitude,axis=2)   # pylint: disable=typecheck
    
    t_fit = (T_peak+2/f_exp.magnitude,T_peak+7/f_exp.magnitude)
    tstart = np.argmin(np.abs(t[np.newaxis, np.newaxis, :]-t_fit[0][:, :, np.newaxis]),axis=2)
    tstop =  np.argmin(np.abs(t[np.newaxis, np.newaxis, :]-t_fit[1][:, :, np.newaxis]),axis=2)
    
    # Derive Sw parameter, normalized
    S_w_orig=_get_Sw(waveforms,tstart,tstop)   # pylint: disable=typecheck

    # Get resonance frequency
    Ha_Rwin_dB = _res_Spec(tstart,tstop,Nfft,waveforms)   # pylint: disable=typecheck
    f_ha,_ = find_peak(Hafreq[0:Nfft2],Ha_Rwin_dB[:,:,0:Nfft2])   # pylint: disable=unsubscriptable-object
    assert isinstance(f_ha, Quantity)
    
    # Derive spectrum of the first reflection        
    t_fitF = (T_peak-2/f_ha.magnitude, T_peak+2/f_ha.magnitude)
    tfstart = np.argmin(np.abs(t[np.newaxis, np.newaxis, :]-t_fitF[0][:, :, np.newaxis]),axis=2)
    tfstop =  np.argmin(np.abs(t[np.newaxis, np.newaxis, :]-t_fitF[1][:, :, np.newaxis]),axis=2)
    
    #Upscaling 
    NfftS=Nfft*10
    waveformsScale=np.zeros([M,N,NfftS])
    waveformsScale[:,:,0:Nfft]=waveforms
    S_Ha = _tukey_Spec(tfstart, tfstop, NfftS, waveformsScale)   # pylint: disable=typecheck
    SHafreq=np.arange(0,NfftS)/NfftS/((dt))
    
    
    return f_ha, S_w_orig, S_Ha, SHafreq

#%%
@guvectorize([(int64, float64, float32, float64, float64,float64[:],complex128[:])], '(),(),(),(),(),(n)->(n)', \
              nopython=True, target='parallel')           
def Mandal_plane_wave(ZfoH: int, Ct:float, iim:float, cim:float, cvp:float, Hafreq: np.ndarray, R: np.ndarray):
    """ 1 D model as described  in Madal however *-1 (Mandal result has a 180 degree phase shift) 
    
    Args:
        ZfoH: the guess for the impedance behind the pipe
        Ct: the guess for the thickness
        iim: the impedance of the fluid inside the pipe
        cim: the impedance of the casing/pipe
        cvp: speed of the casing
        Hafreq: the frequency vector.
        
    Returns:
        R (np.ndarray): Transformation function
    """
    R[:] = (cim-iim)/(iim+cim) \
        - ( (((4*iim*cim)/((iim+cim)**2))*((cim-ZfoH)/(cim+ZfoH))) \
           / (1-(((cim-iim)/(iim+cim))*((cim-ZfoH)/(cim+ZfoH))*np.exp(-1j*2*2*np.pi*Hafreq[:]*Ct/cvp))) \
           * np.exp(-1j*2*2*np.pi*Hafreq[:]*Ct/cvp))

def _ABCD_Z(series: PulseEchoSeries ,S_Ha: np.ndarray ,Ct: Quantity ,S_w_orig: np.ndarray)-> Quantity:
    """ Calculation of function and parameters ABCD as described in Mandal using the 1D model 
    
    Args:
        series: `PulseEchoSeries` containing input waveform data
        S_Ha: Spectrum of the first pulse used in the convolution with the plane wave model result
        Ct: the matrix containing the derived thickness values
        S_w_orig: Sw parameter derived from waveform data
    
    Returns:
        Z: derived impedance
    """
    assert series.casing is not None and series.casing.material is not None \
        and series.casing.material.speed is not None and series.casing.material.impedance is not None
    assert series.inner_material is not None and series.inner_material.impedance is not None
    assert series.casing.thickness_nominal is not None

    ZfoH=Quantity([1,4,8,1,4,8,1,4,8],'MRayl')
    Ctd=Quantity([-0.000635,-0.000635,-0.000635,0,0,0,0.000635,0.000635,0.000635],'m')
    
    f_exp= series.casing.material.speed/(2*series.casing.thickness_nominal)
    dt=1/series.sampling_freq
    t=series.data.t.to_numpy()
    [M,N,Nfft]=S_Ha.shape
    Hafreq=np.arange(0,Nfft)/Nfft/((dt.magnitude))
    
    Ctv=Ctd[np.newaxis,np.newaxis,:] +Ct[:,:,np.newaxis]   
    
    Mat=np.zeros([len(ZfoH),4])
    Mat[:,0]=1
    
    Z=np.zeros([M,N])  
    for i in range(0,M):
        for j in range(0,N):
            try:
                # Ctv=Ctd +Ct[i,j]     
                Mat[:,1]=Ctv[i,j,:].magnitude                    
                # pylint: disable-next=typecheck
                R_HA = Mandal_plane_wave(ZfoH.magnitude,
                                         Ctv[i,j,:].magnitude,
                                         series.inner_material.impedance[i].magnitude,
                                         series.casing.material.impedance.magnitude,
                                         series.casing.material.speed.magnitude,
                                         Hafreq)
                ConvHa = R_HA*S_Ha[np.newaxis,i,j,:]   # conv. of refl_coeff with spectrum of norm. window, and into time domain
                data_mod = np.real(ifft(ConvHa[:,:int(Nfft/2)],Nfft,axis=1))
                T_peak_new = _find_reflection(data_mod,f_exp.magnitude,t[0],dt.magnitude,axis=1)   # pylint: disable=typecheck
                t_fit_new = (T_peak_new+(2/f_exp.magnitude),T_peak_new+(7/f_exp.magnitude))
                tstart = np.argmin(np.abs(t[np.newaxis, :]-t_fit_new[0][:, np.newaxis]),axis=1)
                tstop =  np.argmin(np.abs(t[np.newaxis, :]-t_fit_new[1][:, np.newaxis]),axis=1)
                S_w_mod=_get_Sw(data_mod,tstart,tstop)   # pylint: disable=typecheck
                Mat[:,2]=np.log(S_w_mod)
                Mat[:,3]=Ctv[i,j,:].magnitude*Mat[:,2]
                a0,b0,c0,d0=np.linalg.lstsq(Mat,ZfoH.magnitude,rcond=None)[0]
                Z[i,j]=(a0+b0*Ct[i,j].magnitude+c0*np.log(S_w_orig[i,j])+d0*Ct[i,j].magnitude*np.log(S_w_orig[i,j]))    
            except:   # FIXME: Specify exception type   # pylint: disable=bare-except
                Z[i,j]=None
            
    return Quantity(Z,'MRayl')
