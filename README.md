# HDF to N5 Converter
Command-line converter tool for turning labeled HDF5 data into an [N5](https://github.com/saalfeldlab/n5) label dataset of arbitrary cell size, which allows for parallel writing. In the future, this will probably be how [BIGCAT](https://github.com/saalfeldlab/bigcat) will allow collaborative editing of labels.

## Compile

To compile a "fat jar" with all dependencies added, run:

```
mvn clean compile assembly:single
```

## Usage

```
java -jar target/hdf-n5-converter-0.0.1-SNAPSHOT-jar-with-dependencies.jar [args]
```

#### Arguments

-  `--cellsize`, `-cs`
   Size of cells to use in the output N5 dataset. Default: `64,64,8`
   
-  `--compression`, `-c`
   Compression type to use in output N5 dataset. Default: `RAW`
   
-  `--datasetname`, `--data`, `-d`
   Output dataset name (N5 relative path from group)
   (**Required**)
   
-  `--groupname`, `--group`, `-g`
   Output group name (N5 group)
   (**Required**)
   
-  `--inputfile`, `--input`, `-i`
   Input HDF5 file
   (**Required**)
   
-   `--label`, `-l`
   Path to labels in HDF5 file. Default:`/volumes/labels/neuron_ids`

#### Example

```
java -jar target/hdf-n5-converter-0.0.1-SNAPSHOT-jar-with-dependencies.jar --input ~/sample_A_20160501.hdf --group ~/n5 --data sampleA-fullres --compression GZIP
```

Would write an N5 dataset at `~/n5/sampleA-fullres` from the labels at `/volumes/labels/neuron_ids` in `~/sample_A_20160501.hdf` with GZIP compression (recommended).



**NOTE:** This currently requires [`LabelUtils`](https://github.com/shrucis1/bigcat/blob/85c26f718cae97a133e279f3f3ec1e3ab7eaa73d/src/main/java/bdv/labels/labelset/LabelUtils.java) from the `n5cacheloader2` branch of BIGCAT, which currently is not merged into `master`.
