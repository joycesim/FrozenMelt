# VBRc and related post-processing

The code here loads and process model output and calculates seismic characteristics 
using the VBRc and additional analysis with Python and GNU Octave.

## Requirements and Setup 

### Python 

Python >=3.9 should work, code was written and tested with Python 3.10.11. 

After cloning or downloading this repository, install additional Python packages: 

* Install the Python requirements
* Install (or verify) VBRc  

#### Install Python requirements 

To install the python requirements, 

First make sure pip is up to date with

```shell
pip install --upgrade pip
```

Now install the requirements for the code here, including yt_aspect>0.1.0 (which now includes a general .pvtu loader, not just ASPECT .pvtu)

```shell
pip install -r requirements.txt
```

### GNU Octave and VBRc setup 

While the VBRc ([link](https://vbr-calc.github.io/vbr)) is built for both MATLAB
and GNU Octave, the analysis here relies on GNU Octave. The Python script 
`step_02_process_runs_with_dask.py` will spawn an Octave process for each data file, 
so to use this code as-is, you'll need to install GNU Octave 
(https://octave.org/download). To use MATLAB, you would need to modify 
`ridge_post_proc.main.process_single_run` where it starts `octave`. 

#### Install (or verify) VBRc

Once you have GNU Octave installed,  you will need to install the VBRc. 

If you already have a local installation, you can simply set 
the  `vbrdir` environment variable to point to the top level of the VBRc. 

You also can use `step_01_check_vbrc.py` to: 
* verify your VBRc installation: `python step_01_checkvbrc.py`
* download a copy VBRc v1.1.2: `python step_01_checkvbrc.py 1`

Full installation instructions for the VBRc can be found [here](https://vbr-calc.github.io/vbr/gettingstarted/installation/).

### Additional requirements

Note that the code will generate ~1.8 GB of intermediate outputs so you'll want 
to be sure you've got the disk space.

## Running the analysis 

After installation, run the analysis in three steps. All of the commands below assume that you 
are in the same directory as this README.md file (`FrozenMelt/seismic_post_processing`).

### 1. (optional) Verify VBRc installation

If you haven't run `step_01_checkvbrc.py` yet run it now to make sure the VBRc can be found 

```shell
python step_01_checkvbrc.py
```

### 2. Process model output 

The `step_02_process_runs_with_dask.py` script uses Dask to process model output in parallel. It 
uses 6 single-threaded works by default. To use a different number of workers, provide an extra 
integer argument. For example, for 2 workers:

```shell
python step_02_process_runs_with_dask.py 2
```

To use the default, just run 

```shell
python step_02_process_runs_with_dask.py
```

This will result in ~1.8 GB of VBRc `.mat` output files and other intermediate data in `output/` 

### 3. process the VBRc output 

The following script will load the VBRc output back in to build plots and run 
additional analysis: 

```shell 
python step_03_process_vbrc_output.py
```

## Contents 

Directories and files: 

* `ridge_post_proc`: a python module containing post-processing functions to run the VBRc and make figures
* `data`: the data directory for raw model data (see top level readme for info on fetching data)
* `step_01_check_vbrc.py`: verify VBRc installation and/or download the VBRc
* `step_02_process_runs_with_dask.py`: runs initial processing (calls the VBRc)
* `step_03_process_vbrc_output.py`: load in processed data, make figures
* `run_VBRc.m`: the VBRc driver, called from `ridge_post_proc` (do not call directly)
* `vbr_helper`: extra matlab functions used by `run_VBRc.m`
* `output`: directory that will contain figures and processed output files 
* `figs_for_repo`: folder with figures saved for reference
