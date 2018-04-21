# (BigCAT) Spark Downsampler
Command-line tool for downsampling label data (stored as LabelMultisetTypes in an [N5](https://github.com/saalfeldlab/n5) dataset) with Spark.

## Compile

First install SNAPSHOT dependencies that are not available through remote maven repositories:

 - Clone [imglib2-cache](https://github.com/imglib/imglib2-cache): `git clone https://github.com/imglib/imglib2-cache`
 - Install
```
mvn clean install
```
 - Clone [label-multisets](https://github.com/hanslovsky/label-multisets): `git clone https://github.com/hanslovsky/label-multisets`
 - Install
```
mvn clean install
```
 - Clone [n5-label-multisets](https://github.com/hanslovsky/n5-label-multisets): `git clone https://github.com/hanslovsky/n5-label-multisets`
 - Install
```
mvn clean install
```
 - Finally, clone this repository, and it should compile.
 - To make a "shaded jar" with all dependencies added (local jobs), run:
```
mvn -PfatWithSpark -Denforcer.skip=true clean package
```
 - To make a "shaded jar" with all dependencies except Spark added (cluster jobs), run:
```
mvn -Pfat -Denforcer.skip=true clean package
```

## Downsampling

![Downsampling Operation](/img/downsampling_example.png)

<img align="right" src="/img/labelmultiset_structure.png" height=175 />

Downsampling works by merging groups of `LabelMultisetEntryList`'s:

For instance, in a block of pixels where each list has one entry,
with a count of 1 and a label of either 83 or 64, they will be downsampled
into a single list with 2 entries, one of which will have a label of 83
and the corresponding count, and the other will have a label of 64 and its
corresponding count.

The motivation for this method of downsampling is that it doesn't technically
lose information other than where within a downsampled block each label is.

Most importantly, for sparse labels, this allows to query lists of blocks that contain a specific label id very efficiently. This is very useful for on the fly 3D mesh generation via marching cubes. Also, when converting to a color for visualization purposes, the transition between scale levels when rendering arbitrary 2D slices gets smoother as individual pixels can be assigned colors based on the weighted sums of the contained labels.

## Usage

Run `[SPARK_MASTER=<master>] [JAR=<jar>] ./multisets-downsampler <args>` (bash or compatible shell reuqired). Square brackets indicate optional arguments. If no bash or compatible shell is available, copy and modify the contents of `multisets-downsampler` as required. This script will invoke the main class `bdv.bigcat.spark.SparkDownsampler`:
```bash
java -cp "${JAR}" \
     -Dspark.master="${SPARK_MASTER}" \
     bdv.bigcat.spark.SparkDownsampler \
     $@
```

#### Positional Arguments
- List of downsampling factors per scale level, separated by spaces. Each factor is relatitve to the previous scale level. Factors can should adhere to the format `fX,fY,fZ` for anisotropic downsampling, or simply `f` for isotropic downsampling.

#### Options

- `--block-size`, `-b`
   Chunk size of the target n5 dataset. Downsampling will be parallelized over these blocks. This is optional, for any scale level for which no block size is defined, this defaults to the block size of the previous scale level. Block sizes should adhere to the format `bX,bY,bZ` for anisotropic block sizes, or simply `b` for isotropic block sizes. If no block size is defined at all, this defaults to `64`.

-  `--compression`, `-c`
   Compression type to use in output N5 dataset. *Default*: `{\"type\":\"raw\"}`
   
-  `--dataset`, `-d`
   Multiscale dataset. The first scale level at `dataset/s0` is required to be present unless `-l`/`--link-mipmap-level-zero` is specified.
   
-  `--group`, `--g`
   N5 root
   
-  `-l`, `--link-mipmap-level-zero`
   Create the first scale level as symlink
   
-  `--help`, `--h`
   Show help/usage
   
Note that the `spark.master` property must be set when running as well. See [here](http://spark.apache.org/docs/latest/submitting-applications.html#master-urls) for more information on Spark Master URLs.

#### Example
```bash
N_NODES=${N_NODES:-10}
N_EXECUTORS_PER_NODE=${N_EXECUTORS_PER_NODE:-15}

# N5 group that holds integer type labels
# N5_GROUP="/nrs/saalfeld/hanslovskyp/n5-examples"
N5_GROUP="/nrs/saalfeld/lauritzen/02/example.n5"
# input dataset
DATASET="volumes/labels/multicut_more_features-multisets-256,256,26"

JAR="$HOME/bigcat-spark-downsampler-0.0.1-SNAPSHOT-shaded.jar"
CLASS="bdv.bigcat.spark.SparkDownsampler"

N_EXECUTORS_PER_NODE=$N_EXECUTORS_PER_NODE \
         $HOME/flintstone/flintstone.sh ${N_NODES} $JAR $CLASS \
         -g "${N5_GROUP}" \
         -d "${DATASET}" \
         -b 256,256,52 \
         -b 256,256,104 \
         -b 256,256,208 \
         -m 5 \
         2,2,1 2,2,1 2,2,1 2,2,2 2,2,2 2,2,2 2,2,2 2,2,2 2,2,2 2,2,2
```
