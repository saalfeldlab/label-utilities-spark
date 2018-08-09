package org.janelia.saalfeldlab.label.spark.uniquelabels;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.concurrent.Callable;

import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.label.spark.exception.InvalidDataset;
import org.janelia.saalfeldlab.label.spark.exception.InvalidN5Container;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Parameters;

public class LabelToBlockMappingN5
{

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static class CommandLineParameters implements Callable< Void >
	{

		@Parameters( index = "0", paramLabel = "INPUT_N5", description = "Input N5 container (has to be file system)" )
		private String inputN5;

		@Parameters( index = "1", paramLabel = "INPUT_DATASET", description = "Input dataset. Must be LabelMultisetType or integer type" )
		private String inputDataset;

		@CommandLine.Option(names={"--output-n5", "-o"}, paramLabel = "OUTPUT_N5", description = "Output n5 root (filesystem, defaults to INPUT_N5)")
		private String outputN5;

		@CommandLine.Option(names={"--output-group", "-g"}, paramLabel = "OUTPUT_GROUP", description = "Output group, defaults to 'label-to-block-mapping'")
		private String outputGroup;

		@CommandLine.Option(names={"--step-size", "-s"}, paramLabel = "N5_STEP_SIZE", description = "Store as n5 instead of single file with <N5_STEP_SIZE> entries per block. Has to be larger than 0.")
		private Integer n5StepSize;

		@Override
		public Void call() throws Exception
		{
			if ( this.inputN5 == null ) { throw new InvalidN5Container( "INPUT_N5", this.inputN5 ); }

			if ( this.inputDataset == null ) { throw new InvalidDataset( "INPUT_DATASET", inputDataset ); }

			this.outputN5 = this.outputN5 == null ? this.inputN5 : this.outputN5;

			this.outputGroup = this.outputGroup == null ? "label-to-block-mapping" : this.outputGroup;

			this. n5StepSize = Math.max(this.n5StepSize == null ? 1 : this.n5StepSize, 1);

			final long startTime = System.currentTimeMillis();

			final SparkConf conf = new SparkConf().setAppName( MethodHandles.lookup().lookupClass().getName() );

			try (final JavaSparkContext sc = new JavaSparkContext( conf ))
			{
				LabelToBlockMapping.createMappingWithMultiscaleCheckN5(sc, inputN5, inputDataset, outputN5, outputGroup, n5StepSize);
			}

			final long endTime = System.currentTimeMillis();

			final String formattedTime = DurationFormatUtils.formatDuration( endTime - startTime, "HH:mm:ss.SSS" );
			LOG.info( "Created label to block mapping for " + inputN5 + ":" + inputDataset + " at " + outputGroup + " in " + formattedTime );
			return null;

		}
	}

	public static void run( final String[] args ) throws IOException
	{
		CommandLine.call( new CommandLineParameters(), System.err, args );
	}

	public static void main( final String[] args ) throws IOException
	{
		run( args );
	}


}
