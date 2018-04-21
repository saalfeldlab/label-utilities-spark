# HDF to N5 Converter
Command-line converter tool for turning labeled HDF5 data into an [N5](https://github.com/saalfeldlab/n5) label dataset of arbitrary cell size, which allows for parallel writing. In the future, [BigCAT](https://github.com/saalfeldlab/bigcat) will probably use N5 as the backend for label data.

## Compile

Currently depends on:
https://github.com/hanslovsky/n5-hdf5/tree/reorder-array-attributes

To compile a "fat jar" with all dependencies added, run:

```
mvn -Pfat clean package
```

## Usage

```BASH
N_NODES=${N_NODES:-10}
N_EXECUTORS_PER_NODE=${N_EXECUTORS_PER_NODE:-15}

# N5 group that holds integer type labels
N5_GROUP="/nrs/saalfeld/lauritzen/02/workspace.n5"
# group for output (optional, defaults to ${N5_GROUP})
# N5_GROUP_OUTPUT="/nrs/saalfeld/hanslovskyp/n5-examples"
# N5_GROUP_OUTPUT="/groups/saalfeld/home/hanslovskyp/lauritzen/03/workspace.n5"
N5_GROUP_OUTPUT="/nrs/saalfeld/lauritzen/02/example.n5"

# input dataset
DATASET="filtered/segmentation_old/multicut_more_features"

BLOCKSIZE=${BLOCKSIZE:-64,64,64}

# N5 dataset for result (optional, defaults to 'supervoxels')
TARGET="volumes/labels/multicut_more_features-multisets-${BLOCKSIZE}/s0"
N
JAR="$HOME/hdf-n5-converter-0.0.1-SNAPSHOT-shaded.jar"
CLASS="bdv.bigcat.util.HDFConverter"

N_EXECUTORS_PER_NODE=$N_EXECUTORS_PER_NODE \
         $HOME/flintstone/flintstone.sh ${N_NODES} $JAR $CLASS \
         -g "${N5_GROUP}" \
         -G "${N5_GROUP_OUTPUT}" \
         -d "${DATASET}" \
         -b "${BLOCKSIZE}" \
         "${TARGET}"

```
