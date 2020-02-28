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

- Navigate to .../gm_classifier/gm_classifier/scripts
- Use the script ```run.py```, which has the following options
```
usage: run.py [-h] [--event_list_ffp EVENT_LIST_FFP]
              [--record_list_ffp RECORD_LIST_FFP]
              [--ko_matrices_dir KO_MATRICES_DIR] [--low_memory]
              record_dir model output_ffp

positional arguments:
  record_dir            Root directory for the records, will search for
                        records recursively from here
  model                 Either the path to a saved keras model or the name of
                        an original model (i.e. ['canterbury',
                        'canterbury_wellington']
  output_ffp            Output csv path

optional arguments:
  -h, --help            show this help message and exit
  --event_list_ffp EVENT_LIST_FFP
                        Path to file that lists all events to use (one per
                        line). Note: in order to be able to use event
                        filtering, the path from the record_dir has to include
                        a folder with the event id as its name. Formats of
                        event ids: just a number or XXXXpYYYYYY (where XXXX is
                        a valid year)
  --record_list_ffp RECORD_LIST_FFP
                        Path to file that lists all records to use (one per
                        line)
  --ko_matrices_dir KO_MATRICES_DIR
                        Path to the directory that contains the Konno
                        matrices. Has to be specified if the --low_memory
                        options is used
  --low_memory          If specified will prioritise low memory usage over
                        performance. Requires --ko_matrices_dir to be
                        specified.
```

E.g. ```python run_.py /home/cbs51/code/gm_classifier/data/records/ObservedGroundMotions_GMC canterbury /home/cbs51/code/tmp/gm_classifier/record_output.csv --event_list_ffp /home/cbs51/code/gm_classifier/gm_classifier/my_stuff/event_list.txt --ko_matrices_dir /home/cbs51/code/gm_classifier/data```





