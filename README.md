# GPR Modeling and Processing

This repository contains a collection of Python scripts and notebooks for **simulating, processing, and analyzing Ground Penetrating Radar (GPR) data**. The project combines basic GPR forward modeling with common processing steps (filtering, gain, background removal) and exploratory analysis tools such as migration and hyperbola fitting.

This codebase is **research-oriented and exploratory** rather than a polished software package.

---

## Project Structure

GPR-modeling/
│
├── RoPeR/
│ ├── bandpass.py # Bandpass filtering for GPR traces
│ ├── gain.py # Gain functions for amplitude correction
│ ├── bgr.py # Background removal routines
│ ├── energy_eq.py # Energy equalization utilities
│ ├── prestitch.py # Preprocessing / trace stitching tools
│ └── init.py
│
├── Simulating-GPR/
│ ├── gprsim.py # GPR forward simulation code
│ ├── migration.py # Migration routines for GPR data
│ ├── Hyperbola_Fitting.ipynb
│ ├── gprsim.ipynb
│ └── temp.ipynb
│
├── README.md
└── .gitignore

---

## Overview

The repository is split into two main components:

### 1. RoPeR (Processing Utilities)

`RoPeR/` contains reusable Python functions for **GPR preprocessing**, including:

- Bandpass filtering
- Time-dependent gain
- Background removal
- Energy equalization
- Pre-stitch preprocessing of radar traces
- post-processing migation algorithms

These tools are intended to be modular and imported into notebooks or scripts.

### 2. Simulating-GPR (Modeling & Analysis)

`Simulating-GPR/` contains scripts for:

- Forward simulation of GPR wave propagation
- Exploring synthetic radargrams
- Applying migration algorithms
- Hyperbola fitting for velocity estimation

The notebooks are primarily exploratory and demonstrate how the simulation and processing tools can be used together.

---

## Requirements

This project is written in **Python 3** and primarily relies on standard scientific Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `jupyter`

Some scripts may assume additional packages depending on the workflow.

---

## Usage

### Running Simulations

To run a basic GPR simulation:

```bash
python Simulating-GPR/gprsim.py

