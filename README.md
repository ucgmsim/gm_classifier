## Ground Motion Classifier 

### Description

This repository contains the code needed to use the GM quality estimation model from the paper "A deep-learning-based model for quality assessment of earthquake-induced ground motion records". Additionally it also contains the labels used to train the model, and a script to re-train it if desired.

The model estimates the record quality and minimum usable frequency of each component with

- a quality score between 0-1 and
- and minimum usable frequency between 0.1-10 Hz

This means that for a 3-component record 6 outputs are produced, which can then be easily mapped to a binary classification via a user defined mapping function, making the outputs useful for a wide range of applications.

For the full details see the paper.



### Installation

Requires python 3.8 or 3.9, should work with newer versions as well, but has not been tested

- Clone the repository 
  `git clone git@github.com:ucgmsim/gm_classifier.git`
- Install requirements
  `pip install -r requirements.txt`



### Usage

Getting estimation is a two step process, 1) the features are extracted from the acceleration time-series and 2) features are passed into the model to estimate quality and minimum usable frequency



#### Feature extraction

Feature extraction is done using the `extract_features.py` script, which takes the following arguments

```shell
positional arguments:
  output_dir            Path to the output directory
  record_dir            Root directory for the records, will search for V1A or mseed records recursively
                        from here
  {V1A,mseed}           Format of the records, either V1A or mseed

optional arguments:
  -h, --help            show this help message and exit
  --output_prefix OUTPUT_PREFIX
                        Prefix for the output files
  --record_list_ffp RECORD_LIST_FFP
                        Path to file that lists all records to use (one per line)
  --ko_matrices_dir KO_MATRICES_DIR
                        Path to the directory that contains the Konno matrices. Has to be specified if
                        the --low_memory options is used
  --low_memory          If specified will prioritise low memory usage over performance. Requires
                        --ko_matrices_dir to be specified.
```

The `--record_list_ffp` options was added to allow compute features for a subset of records found in the `record_dir` folder (and sub-folders), where the `record_id` is the name of the record file without the extension, e.g. for `20170102_131402_DREZ_10_EH.mseed` the `record_id` is `20170102_131402_DREZ_10_EH`  

  

The `--ko_matrices_dir` option allows specifying of the directory that contains the pre-computed Konno-Ohmachi matrices, for more details see below. It is recommended to pre-compute these.
The `--low_memory` option can only be used when the Konno-Ohmachi matrices directory has been specified, and reduces the total memory usage at the cost of performance. If your machine has 16GB  or more RAM then this is not required (unless larger Konno-Ohmachi matrice sizes are specified manually, as per below)



##### Konno-Ohmachi matrices

Feature computation uses Konno-Ohmachi matrices for smoothing of the Fourier amplitude spectra, and it is recommended to pre-compute (but not required, in which case this is done on the fly). This can be done using the `gen_konno_matrices.py` script, which just takes an output directory as input.

By default Konno-Ohmachi matrices of the following sizes `[1024, 2048, 4096, 8192, 16384, 32768]` are computed, supporting records with duration of up to ~327s at dt=0.005. If this is not sufficient, then the Konno-Ohmachi matrice sizes can be overwritten with the `KO_MATRIX_SIZES` environment variable, e.g.  `export KO_MATRIX_SIZES=1024,2048,4096,8192,16384,32768,65536,131072`
If the environment variable is set for generation of the matrices, then it also has to be set when running the feature extraction, otherwise the `extract_features.py` script will use the default sizes.

Note: The larger Konno-Ohmachi matrices require significantly more computation time and memory



#### Quality & minimum usable frequency estimation

Once the features have been computed, the `predict.py` script, which takes the following arguments

```shell
positional arguments:
  features_dir          Path of directory that contains the feature files
  output_ffp            File path for output csv

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Path of directory that contains the GMC model, defaults to the model included in
                        the repository
```

The resulting comma-separated value file has the following format, with each row corresponding to a record component

```
record_id,score_mean,score_std,fmin_mean,fmin_std,multi_mean,multi_std,record,component
20180708_234345_CECS_20_X,0.986889,0.0178299,0.274238,0.0974789,0.004214,0.00834436,20180708_234345_CECS_20,X
20180708_234347_MOLS_20_X,0.996289,0.00881992,0.222371,0.0942545,0.000526558,0.00106552,20180708_234347_MOLS_20,X
20180708_234403_BWRS_20_X,0.365711,0.130317,0.468655,0.195347,0.0271395,0.0221378,20180708_234403_BWRS_20,X
20180708_234400_SEDS_20_X,0.69397,0.283386,0.396147,0.118394,0.292748,0.303592,20180708_234400_SEDS_20,X
20180708_234402_MATS_20_X,0.946903,0.0473075,0.486201,0.135772,0.00190778,0.0041504,20180708_234402_MATS_20,X
```



#### Usage Example

```shell
# Update Konno matrices sizes (optional, not needed in most cases)
export KO_MATRIX_SIZES=1024,2048,4096,8192,16384,32768,65536,131072

# Generate Konno matrices
python gen_konno_matrices.py /path/to/konno-matrices-directory 							

# Extract features
python extract_features.py /path/to/feature-output-directory /path/to/record-root-directorey mseed --ko_matrices_dir /path/to/konno-matrices-directory

# Get quality & mimium usable frequency estimations
python predict.py /path/to/feature-output-directory /path/to/output-csv-file
```









