# FrozenMelt

This repository contains code for reproducing figures for the following publication: 

> Sim, S. J., Yu, T. Y., Havlin, C. Persistent heterogeneities in the oceanic lithosphere due to differential freezing beneath ridges

The code is a mix of MATLAB (or GNU Octave) and Python code. 

The repository is divided into two main directories:

* `post_processing` : reproduces figure 3 with MATLAB
* `seismic_post_processing`: reproduces seismic analysis with a mix of Python and GNU Octave

To run the code, you will first want to fetch the data.

## Fetching the data 

The required data are stored in 

> Sim, S. J., Yu, T.-Y., & Havlin, C. (2023). Frozen Melt Data (0.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10288494

to fetch the data and put it in the correct directories expected by the code, there
are some helpful scripts in this repository. 

If you have Python (which you will want for the `seismic_post_processing`, 
see below), then run

```commandline
$ pip install pooch
$ python fetch_data_with_pooch.py
```

If you do not have Python (and only want to run the code in `post_processing/`), you can 
instead try the following bash script that uses `curl`,

```commandline
$ ./fetch_data_with_curl.sh
```

If none of the above work, or if you prefer to manually fetch the data:

1. Download a zip file, `FrozenMeltData.zip` from https://doi.org/10.5281/zenodo.10288494
2. Unpack all the archives 
3. Move the contents of `FrozenMeltData/PostProcessed/` to `post_processing/`
4. Move the contents of `FrozenMeltData/VTU/` to `seismic_post_processing/data`

See https://doi.org/10.5281/zenodo.10288494 for a description of the data.

## Running the code 

### Running code in `post_processing`

**Requirements**: MATLAB or GNU Octave

**Instructions**: To run this code, simply change into the `post_processing` 
directory and run `Plotting_Figure_3.m` in MATLAB or GNU Octave.

### Running code in  `seismic_post_processing`

**Requirements**: GNU Octave and the VBRc, Python>=3.9 (and the packages in `seismic_post_processing/requirements.txt`)
**Instructions**: Detailed instructions are provided in `seismic_post_processing/README.md`.

Note that `seismic_post_processing` includes a VBRc installation script, see `seismic_post_processing/README.md`.