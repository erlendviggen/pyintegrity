# Pyintegrity overview

Pyintegrity is an open-source Python library for processing acoustic well integrity log data, in particular ultrasonic pulse-echo data. With Pyintegrity, you can load well logs that contain the original recorded waveforms and process them using any of the implemented processing algorithms. You can also experiment with these algorithms on real or simulated data, or use Pyintegrity as a framework for developing and testing new algorithms.

You can find a more detailed introduction to Pyintegrity in our article *Pyintegrity: An Open-Source Toolbox for Processing Ultrasonic Pulse-Echo Well Integrity Log Data*, presented at the SPE Norway Subsurface Conference 2024, SPE-218476-MS.

Pyintegrity was originally developed by NTNU Researcher [Erlend Magnus Viggen](https://erlend-viggen.no/) (most of the framework, W2/W1 algorithm) and SINTEF Research Scientist [Anja Diez](https://www.sintef.no/en/all-employees/employee/anja.diez/) (simulation framework, all other algorithms) in a project funded by [The Research Council of Norway](https://www.forskningsradet.no/en/) through the [Centre for Innovative Ultrasound Solutions](https://www.ntnu.edu/cius) (grant no. 237887).

Pyintegrity is made available under an [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).



## Getting started

To download and install Pyintegrity in a conda environment named `pyintegrity`, do the following:

```
git clone https://github.com/erlendviggen/pyintegrity.git
cd pyintegrity
conda env create -f environment.yml
conda activate pyintegrity
pip install -e .
```

The included Jupyter Notebook `Pyintegrity demo.ipynb` provides a demonstration of Pyintegrity by reproducing the results of the article.



## Currently implemented

### Processing algorithms

Pyintegrity currently implements four ultrasonic pulse-echo processing algorithms:

1. [W2/W1 algorithm](api-pulseecho-processing.md#w2w1-algorithm) ([Havira 1981](https://patents.google.com/patent/US4255798A/en))
2. [T<sup>3</sup> algorithm](api-pulseecho-processing.md#t3-algorithm) ([Hayman et al. 1991](https://onepetro.org/SPWLAALS/proceedings-abstract/SPWLA-1991/All-SPWLA-1991/18954))
3. [ABCD algorithm](api-pulseecho-processing.md#abcd-algorithm) ([Mandal and Standley 2000](https://patents.google.com/patent/US6041861A/en))
4. [L1 algorithm](api-pulseecho-processing.md#l1-algorithm) ([Tello 2010](https://patents.google.com/patent/US7755973B2/en))
    * Note that the L1 algorithm is still patented in the USA and Canada. Use the Pyintegrity L1 implementation on your own legal responsibility.


### Tool data extraction

Because different logging tools from different vendors store their data in different ways, functions to extract tool data from DLIS files must be tailor-made for each tool. Because of a general lack of public documentation, [gaining an understanding about how a new tool's data is structured can require considerable time and effort](https://www.researchgate.net/publication/340645995_Getting_started_with_acoustic_well_log_data_using_the_dlisio_Python_library_on_the_Volve_Data_Village_dataset). Therefore, Pyintegrity currently only has facilities to read log data from tools that the developers have had access to. Currently, Pyintegrity is able to automatically extract data from the log files of:

1. [Ultrasonic Imager Tool (USIT)](https://www.slb.com/products-and-services/innovating-in-oil-and-gas/drilling/drilling-fluids-and-well-cementing/well-cementing/cement-evaluation/usi-ultrasonic-imager) (including the pulse-echo component of the [Isolation Scanner Tool](https://www.slb.com/products-and-services/innovating-in-oil-and-gas/drilling/drilling-fluids-and-well-cementing/well-cementing/cement-evaluation/isolation-scanner)).

Pyintegrity users that have access to and knowledge about other tools' data can write their own code to convert the stored waveform data into [Pyintegrity's data structures](api-pulseecho-data.md#pyintegrity.pulseecho.series.PulseEchoSeries) for processing.


### Simulated data

Pyintegrity facilitates reading simulated measurements from the public dataset [*Simulated ultrasonic pulse-echo well-integrity dataset*](https://data.mendeley.com/datasets/3bs65nzpv2/) by Diez, Viggen, and Johansen.



## Pyintegrity code principles

* Object oriented
* Type hinted
* Google-style docstrings
* Uses the [pint](https://github.com/hgrecco/pint/) (and [pint-xarray](https://github.com/xarray-contrib/pint-xarray)) library to attach physical units to numbers and arrays
* Uses the [Numba](https://numba.pydata.org/) library (and its [Rocket-FFT](https://github.com/styfenschaer/rocket-fft) extension) to accelerate performance-critical code
* Uses the [dlisio](https://github.com/equinor/dlisio) library for reading log files in the DLIS format
