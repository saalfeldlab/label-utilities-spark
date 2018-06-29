# Spark Utilities for Label Data

Utilities for [label multisets](https://github.com/saalfeldlab/n5-label-multisets). Currently available:
 - **convert** integer type n5 dataset into label multiset dataset
 - **downsample** label multiset n5 dataset
 - **extract-unique-labels** from a label dataset
 - **downsample-unique-labels** create mip-map for unique labels
 - **label-to-block-mapping** generate a mapping from label to containing blocks for a dataset of unique label lists

## Compile

To compile for use on local machine, run
```bash
mvn -Denforcer.skip=true -PfatWithSpark clean package
```

To compile for deployment on the Janelia cluster, run
```bash
mvn -Denforcer.skip=true -Pfat clean package
```

## Run
To run locally, run
```bash
java \
    -Dspark.master=local[*] \
    -jar target/label-utilities-spark-0.1.0-SNAPSHOT-shaded.jar COMMAND [--help/-h] [COMMAND ARG...]
```

