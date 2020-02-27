# GM record classifier

Implementation of [Bellagamba et al. (2019)](https://journals.sagepub.com/doi/full/10.1193/122118EQS292M) ([source code](https://github.com/xavierbellagamba/GroundMotionRecordClassifier))

Currently supports using the original models (Canterbury & CanterburyWellingtong) from either a csv file with the metrics
or a set of GeoNet record files.

Also allows extraction of metrics from a set of GeoNet record files into a csv files.

### Setup/Installation
- Clone the repo (i.e. ```git clone git@github.com:ucgmsim/gm_classifier.git``)
- Install using pip, i.e. ```pip install -e ./gm_classifier```

### Classification from feature/metric csv files
- Navigate to .../gm_classifier/gm_classifier/scripts
- Use the script ```run_predict.py```, which has the following options 
```
usage: run_predict.py [-h] model input_data_ffp output_ffp

positional arguments:
  model           Either the path to a saved keras model or the name of an
                  original model (i.e. ['canterbury', 'canterbury_wellington']
  input_data_ffp  Path to the csv with the input data
  output_ffp      Output csv path

optional arguments:
  -h, --help      show this help message and exit
```

E.g. ```python run_predict.py canterbury ../tests/benchmark_tests/original_models/bench_canterbury_data.csv /home/cbs51/code/tmp/output.csv```

### Classification from set of GeoNet records








