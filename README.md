# Pyintegrity

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


## Documentation

Pyintegrity's documentation can be found in the project's GitHub Pages.
