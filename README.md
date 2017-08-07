# (BigCAT) Spark Downsampler
Command-line tool for downsampling label data (stored as LabelMultisetTypes in an [N5](https://github.com/saalfeldlab/n5) dataset) with Spark.

## Compile

Because this repository uses a branch of [BigCAT](https://github.com/shrucis1/bigcat) that is not currently merged into master, to compile from source you will first have to check out that branch (which requires the latest version of N5, which also needs to be compiled).

Clone [N5](https://github.com/saalfeldlab/n5) to any location

Use Maven to install N5 1.1.4-SNAPSHOT into your local repository

```
mvn clean install
```

Then, clone `shrucis1/bigcat` to a location of your choice.

Switch to the `n5cacheloader` branch with the necessary changes.

```
git checkout n5cacheloader
```

Use Maven to install this branch of BigCAT into your local repository

```
mvn clean install
```

Finally, clone this repository, and it should compile.

To make a "fat jar" with all dependencies added, run:

```
mvn clean compile assembly:single
```
