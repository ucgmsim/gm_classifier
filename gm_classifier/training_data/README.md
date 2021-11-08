### Columns description for labels.csv

`score_x`, `score_y`, `score_z`: Component quality score  
`fmin_x`, `fmin_y`, `fmin_z`: Component minimum usable frequency  
`multi`: Flag indicating if the record contains multiple earthquakes that are visible in the PhaseNet output  
`malf`: Flag indicating if the record is considered contains malfunctions
`good_drop`: Normal record (i.e. not malfunctioned or multiple earthquakes) that was dropped due to a bad P-wave pick  
`multi_drop`: Multiple earthquake record that was dropped due to PhaseNet not detecting the multiple earthquakes
`malf_drop`: Malfunctioned record that was dropped due to feature limitations (i.e. no features to detect malfunction)
`other_drop`: Outlier records that were dropped  
`investigate`/`processed`: Columns from labelling process, ignore. Only included as required by the training code.

