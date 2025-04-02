# RallyClipper

This repository handles the automatic clipping of long (~1 hour) raw table tennis match videos into shorter clips of individual rallies.

## 📁 Folder Structure

- `matches/` — This folder should contain the raw `.mp4` files of full-length table tennis matches.
- `outputs/` — An empty folder where the extracted rally clips will be saved.

## 🚀 Getting Started

1. **Install the Conda environment**

Make sure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Then run:

```bash
conda env create -f env.yml

