# GM record classifier

Implementation of [Bellagamba et al. (2019)](https://journals.sagepub.com/doi/full/10.1193/122118EQS292M) ([source code](https://github.com/xavierbellagamba/GroundMotionRecordClassifier))

Currently supports using the original models (Canterbury & CanterburyWellingtong) from either a csv file with the metrics
or a set of GeoNet record files.

Also allows extraction of metrics from a set of GeoNet record files into a csv files.

### Setup/Installation
- Clone the repo (i.e. ```git clone git@github.com:ucgmsim/gm_classifier.git``)
- Install using pip, i.e. ```pip install -e ./gm_classifier```

### Classification from feature/metric csv files
- Use the script ```run_predict.py```, which has the following options 
```
usage: run_predict.py [-h] model input_data_ffp output_ffp

positional arguments:
  model           Either directory of saved model (and the pre-processing
                  information) or the name of an original model (i.e.
                  ['canterbury', 'canterbury_wellington']
  input_data_ffp  Path to the csv with the input data
  output_ffp      Output csv path

optional arguments:
  -h, --help      show this help message and exit
```

E.g. ```python run_predict.py canterbury ../tests/benchmark_tests/original_models/bench_canterbury_data.csv /home/cbs51/code/tmp/output.csv```

### Classification from set of GeoNet records

- Use the script ```run.py```, which has the following options
```
usage: run.py [-h] [--event_list_ffp EVENT_LIST_FFP]
              [--record_list_ffp RECORD_LIST_FFP]
              [--ko_matrices_dir KO_MATRICES_DIR] [--low_memory]
              record_dir model output_ffp

positional arguments:
  record_dir            Root directory for the records, will search for
                        records recursively from here
  model                 Either directory of saved model (and the pre-
                        processing information) or the name of an original
                        model (i.e. ['canterbury', 'canterbury_wellington']
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

### Feature extraction
- Feature extraction is done using the extract_features.py script, which has the following options
```
usage: extract_features.py [-h] [--event_list_ffp EVENT_LIST_FFP]
                           [--record_list_ffp RECORD_LIST_FFP]
                           [--ko_matrices_dir KO_MATRICES_DIR] [--low_memory]
                           output_ffp record_dir

positional arguments:
  output_ffp            File path for the resulting csv file
  record_dir            Root directory for the records, will search for
                        records recursively from here

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
Note:
- Using the low_memory options will slow feature extraction down significantly, especially as the number of records gets larger
- This is a rather slow process (obviously depending on number of records) with high CPU and memory usage, so should ideally 
be run on a reasonably decent machine. 

### Training

- Simply training of a model can be done using the ```train_model.py``` script, which takes the following options
```
usage: train_model.py [-h] [--record_list_ffp RECORD_LIST_FFP]
                      [--val_split VAL_SPLIT]
                      output_dir features_ffp label_ffp config_ffp

positional arguments:
  output_dir            Output directory
  features_ffp          csv file with all the features, as generated by the
                        'extract_features' script
  label_ffp             CSV file with the scores for each record, required
                        columns: ['record_id', 'score']
  config_ffp            Config file, that contains model and training details

optional arguments:
  -h, --help            show this help message and exit
  --record_list_ffp RECORD_LIST_FFP
                        Path to file that lists all records to use (one per
                        line)
  --val_split VAL_SPLIT
                        The proportion of the labelled data to use for
                        validation
```  

E.g. ```python train_model.py /home/cbs51/code/tmp/gm_classifier/train_output_test /home/cbs51/code/gm_classifier/data/records/ObservedGroundMotions_GMC_features.csv /home/cbs51/code/gm_classifier/data/records/ObservedGroundMotions_GMC_labels.csv /home/cbs51/code/gm_classifier/gm_classifier/scripts/train_config.json --val_split 0.1```

- Training can also be done easily in custom scripts by using the "train" function in the training module, example script:
```python
import json

import pandas as pd

import gm_classifier as gm

# Specify the input data
features_ffp = "/home/cbs51/code/gm_classifier/data/records/ObservedGroundMotions_GMC_features.csv"
label_ffp = "/home/cbs51/code/gm_classifier/data/records/ObservedGroundMotions_GMC_labels.csv"

# Training output dir
output_dir = "/home/cbs51/code/tmp/gm_classifier/train_output_test"

# Load the training config
config_ffp = "/home/cbs51/code/gm_classifier/gm_classifier/scripts/train_config.json"
with open(config_ffp, "r") as f:
    config = json.load(f)

# Load the labelled data
features_df = pd.read_csv(features_ffp, index_col="record_id")
label_df = pd.read_csv(label_ffp, index_col="record_id")

# Run the training
gm.training.train(output_dir, features_df, label_df, config, val_split=0.1)

# Classify
result_df = gm.classify.classify(output_dir, features_df)
```





