# Spark Utilities for Label Data

Utilities for [label multisets](https://github.com/saalfeldlab/n5-label-multisets). Currently available:
 - **convert** integer type n5 dataset into label multiset dataset
 - **Downsample** label multiset n5 dataset

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

