#!/bin/bash

# http://picocli.info/autocomplete.html

java -cp $HOME/.m2/repository/info/picocli/picocli/2.3.0/picocli-2.3.0.jar:target/bigcat-spark-downsampler-0.0.1-SNAPSHOT-shaded.jar \
     picocli.AutoComplete \
     -n multisets-downsampler \
     'bdv.bigcat.spark.SparkDownsampler$CommandLineParameters'
