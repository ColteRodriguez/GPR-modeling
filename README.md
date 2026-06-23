## GPR Modeling and Processing

### Explaination and Purpose

This repository contains a collection of Python scripts and notebooks for simulating, processing, and analyzing Ground Penetrating Radar (GPR) data. The project combines basic GPR forward modeling with common processing steps (filtering, gain, background removal) and exploratory analysis tools such as migration and hyperbola fitting.

**Note on University Honor Code**: This repository is associated with my undergraduate senior thesis at Princeton University, accesible [here](https://dataspace.princeton.edu/handle/88435/dsp019c67wm88m) titled "WAVES ON MARS:
IMAGING SUBSURFACE SEDIMENTARY STRUCTURES IN UTOPIA PLANITIA AND IMPLICATIONS ON THE RETREAT OF AN ANCIENT OCEAN" (I'm also happy to send a copy upon email request). It is important to note that changes to this repository made after April 28, 2026 do not represent retroactive changes to this work, and instead represent my efforts at polishing and documenting the tools should anyone be interested in using them in the future. Commits made after the original submission of the thesis represent my reorienting this project towards the broader science community rather than the small committee who evaluated my original submission. In accordance with university regulations, the repository referenced in the data availability statement of the submitted manuscript is still accesible via a past commit. 

This Readme is neither full code documentation or explaination of scientific methods used. The manuscript fulfills the latter and the former is uncessesary as most code is not organized as API or library --- one should be able to simply download and run the code sequentially with few issues outside of possible package installation/version conflicts. Most sections will likely be pretty light except for "Quantifying Migration Uncertainty," as the simulations, at least I've found, are quite useful and might be a nice tool for others to use or modify. 

**TLDR** The target audience for this repository is geoscientitsts with, like me, some knowledge of geophysics (specifically methods in signal processing of data from seismic and GPR surveys). Most of the code here is just implementations or visualizations of  _relatively_ simple geophysics concepts, so I won't be explainaing any methods in detail textbook style. I will, however, explain how to generate FDTD simulations as I see this being the most relevant tool for most readers.

* [1. RoPeR Preprocessing Tools](#1-RoPeR-Preprocessing-Tools)
* [2. Sections](#2-Sections)
    * [2.1 Methods](#21-sub-section)
        * [2.1.1 GPR Fundamentals](#21-sub-section)
        * [2.1.2 Hyperbola Fitting to Estimate Velocity](#21-sub-section)
        * [2.1.3 Quantifying Migration Uncertainty](#21-sub-section)
            * [2.1.3.1 Developing a Ground-Truth Radargram](#21-sub-section)
            * [2.1.3.1 Testing Migration Algorithms](#21-sub-section)
        * [2.1.4 Analyzing Migrated Sections (dip)](#21-sub-section)
    * [2.2 Results: Application to Martian Data](#21-sub-section)
        * [2.2.1 RoPeR Data Pre-Migration Signal Processing](#21-sub-section)
        * [2.2.2 Determining a Velocity Model for Utopia Planitia](#21-sub-section)
        * [2.2.3 RoPeR Data Migration](#21-sub-section)
* [3. Simulating GPR: Libraries to Generate and Migrate Simulated Data ](#2-deep-dive-into-code)


## 1. RoPeR Preprocessing Tools
The RoPeR folder contains first-principles implementations of methods in wavefield preprocessing. 

- **Bandpass filtering** is a simple masking of the data in time-frequency space using np.fft, 
- **gain** is either linear, AGC, or exponential (the optimal gain parameter is dicsussed in the "RoPeR Data Pre-Migration Signal Processing" module --- just minimizing the log variance in the signal stack), and 
- **prestitch.py** handles project specific operations e.g. correctly alligning multiple separate scans. 
- **Lateral energy equalization** is also somewhat project specific as it smooths the signal amplitude along separate scan boundaries. It's exactly what it sounds like, though I will keep this breif as pseudocode and a much more rigorous explaination is given in the manuscript supplemental. 
- **Background removal** implements simple tracewise-mean subtraction, window-trace mean subtraction and something called eigen-background removal --- a cool method from Al-Nuaimy (2010), but one I never ended up using .

## 2. Sections
The intention here was to have a folder associated with each heading of the manuscript so that someone could, as they're reading, understand exactly how methods were implemented and literally reproduce the figures themselves. It might have been a bit much in reprospect, but I liked the attempt at full transparency. Instead of just giving the data and leaving us to write to analysis code ourselves, it is given completely, so that comments and critiques can be made on the actual, original work.

### 2.1 Methods
So many methods here. Lots of documentation is given in-line or in md cells in the ipynb itself, so I'll keep this brief.

#### 2.1.1 GPR Fundamentals
This notebook implements a simple ray-casting, infinite-permitivity reflector GPR simulation for the purpose of explaining how GPR and CMP seismic survey do not image the physical subsurface, rather the recieved wavefeld is a time-smeared "echo" of it. In the simplest case where the subsurface is completely homogenous, a single point reflector (maybe a small rock or pipe) is imaged as a hyperbola --- also called a Normal Moveout. The ray-theoretical, geometric interpretation of the survey may be used to return this hyperbola to its true position, this process of Normal Moveout Correction suffers greatly from unknowns (What is the velocity of radio raves in the subsurface? What is the position of the point reflector along the scan path? What if there exists noise from electronics or other geologic heterogeneities?). We conclude this section asking ourselves: How can we automate Normal Moveout Correction and minimize these uncertainties? The big overarching question is: How do we solve the **Inverse problem** --- That is, how do we take the recieved wavefield, a time-smeared echo of the subsurface, and turn it into an image of the true position of the stratigraphy?

#### Hyperbola Fitting to Estimate Velocity
How do radio waves move through the subsurface? Much like sound waves in a pool, electromagnetic waves travel at different speeds through different media. Albeit, this results from soil properties of dielectric permitivity and magnetic permeability which, while affected by compaction, saturation, and composition, are slightly more complicated, dare I say, than the simple density parameter for sound wave velocity. When emitted radar waves encounter a permittivity boundary (for example, the water inside a pipe burried in a sandy medium), some of the energy gets reflected back to the GPR device while some travels through with a different velocity. Essentially, **imaging** the subsurface reduces to mapping these boundaries, and determining the propogation velocity of their bounded regions. The wave velocity can be estimated from the obliquity of the hyberolae that "spawn out" from point-refelctors. This can be done by a simple rearrangement of the hyperbola equations (3 and 4r in the manuscript).

These hyberbola can be traced by hand, but more systematic methods exist. I found Column Connection Clustering a hopeful solution, and, with a lot of **disclosed** help from Claude to interpret the pseudocode in Barkataki et. al. (2023), I implemented it here. At such low frequency GPR, and low sampling rate, the hyperbola are 1, 2, maybe 3 pixels (if we're lucky) per column, so I 


### 2.1 Sub-section
More content...