# Pulse-echo processing

This page describes the interface to the processing algorithms implemented as separate submodules in Pyintegrity:

* [W2/W1 algorithm](api-pulseecho-processing.md#w2w1-algorithm)
* [T3 algorithm](api-pulseecho-processing.md#t3-algorithm)
* [ABCD algorithm](api-pulseecho-processing.md#abcd-algorithm)
* [L1 algorithm](api-pulseecho-processing.md#l1-algorithm)

While the details of the algorithms are quite different, they all have a common interface. Each algorithm submodule includes a function `process_[algorithmname]` that accepts a `PulseEchoSeries` object (and in many cases other arguments as well) and returns an object of a `ProcessingResult` subclass:

::: pyintegrity.pulseecho.processing.result.ProcessingResult


## W2/W1 algorithm

::: pyintegrity.pulseecho.processing.w2w1
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_root_members_full_path: true


## T<sup>3</sup> algorithm

::: pyintegrity.pulseecho.processing.t3
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_root_members_full_path: true


## ABCD algorithm

::: pyintegrity.pulseecho.processing.abcd
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_root_members_full_path: true


## L1 algorithm

::: pyintegrity.pulseecho.processing.l1
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_root_members_full_path: true
