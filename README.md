# HDF to N5 Converter
Command-line converter tool for turning labeled HDF5 data into an N5 label dataset of arbitrary cell size, which allows for parallel writing. In the future, this will probably be how [BIGCAT](https://github.com/saalfeldlab/bigcat) will allow collaborative editing of labels.

## Compile

To compile a "fat jar" with all dependencies added, run:

```
mvn clean compile assembly:single
```

## Usage

```
java -jar target/hdf-n5-converter-0.0.1-SNAPSHOT-jar-with-dependencies.jar [args]
```

**NOTE:** This currently requires [`LabelUtils`](https://github.com/shrucis1/bigcat/blob/85c26f718cae97a133e279f3f3ec1e3ab7eaa73d/src/main/java/bdv/labels/labelset/LabelUtils.java) from the `n5cacheloader2` branch of BIGCAT, which currently is not merged into `master`.
