package org.janelia.saalfeldlab.label.spark.uniquelabels;

import java.io.File;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.label.spark.N5Helpers;
import org.janelia.saalfeldlab.label.spark.exception.InvalidDataset;
import org.janelia.saalfeldlab.label.spark.exception.InvalidN5Container;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.spark_project.guava.io.Files;

import net.imglib2.algorithm.util.Grids;
import net.imglib2.util.Intervals;
import picocli.CommandLine;
import picocli.CommandLine.Parameters;
import scala.Tuple2;

public class LabelToBlockMapping
{

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static class CommandLineParameters implements Callable< Void >
	{

		@Parameters( index = "0", paramLabel = "INPUT_N5", description = "Input N5 container" )
		private String inputN5;

		@Parameters( index = "1", paramLabel = "INPUT_DATASET", description = "Input dataset. Must be LabelMultisetType or integer type" )
		private String inputDataset;

		@Parameters( index = "2", paramLabel = "OUTPUT_DIRECTORY", description = "Output directory" )
		private String outputDirectory;

		@Override
		public Void call() throws Exception
		{
			if ( this.inputN5 == null ) { throw new InvalidN5Container( "INPUT_N5", this.inputN5 ); }

			if ( this.inputDataset == null ) { throw new InvalidDataset( "INPUT_DATASET", inputDataset ); }

			final long startTime = System.currentTimeMillis();

			final SparkConf conf = new SparkConf().setAppName( MethodHandles.lookup().lookupClass().getName() );

			try (final JavaSparkContext sc = new JavaSparkContext( conf ))
			{
				createMappingWithMultiscaleCheck( sc, inputN5, inputDataset, outputDirectory );
			}

			final long endTime = System.currentTimeMillis();

			final String formattedTime = DurationFormatUtils.formatDuration( endTime - startTime, "HH:mm:ss.SSS" );
			LOG.info( "Created label to block mapping for " + inputN5 + ":" + inputDataset + " at " + outputDirectory + " in " + formattedTime );
			return null;

		}
	}

	public static void run( final String[] args ) throws IOException
	{
		CommandLine.call( new CommandLineParameters(), System.err, args );
	}

	public static final void createMappingWithMultiscaleCheck(
			final JavaSparkContext sc,
			final String inputN5,
			final String inputDataset,
			final String outputDirectory ) throws IOException
	{
		final N5Reader reader = N5Helpers.n5Reader( inputN5, N5Helpers.DEFAULT_BLOCK_SIZE );
		if ( N5Helpers.isMultiScale( reader, inputDataset ) )
		{
			for ( final String scaleDataset : N5Helpers.listScaleDatasets( reader, inputDataset ) )
			{
				LOG.info( "Creating mapping for scale dataset {} in group {} of n5 container {} at target {}", scaleDataset, inputDataset, inputN5, outputDirectory );
				createMapping( sc, inputN5, inputDataset + "/" + scaleDataset, Paths.get( outputDirectory, scaleDataset ).toAbsolutePath().toString() );
			}
		}
		else
		{
			LOG.info( "Creating mapping for dataset {} of n5 container {} at target {}", inputDataset, inputN5, outputDirectory );
			createMapping( sc, inputN5, inputDataset, outputDirectory );
		}
	}

	private static final void createMapping(
			final JavaSparkContext sc,
			final String inputN5,
			final String inputDataset,
			final String outputDirectory ) throws IOException
	{
		final N5Reader reader = N5Helpers.n5Reader( inputN5 );
		final DatasetAttributes inputAttributes = reader.getDatasetAttributes( inputDataset );
		final int[] blockSize = inputAttributes.getBlockSize();
		final long[] dims = inputAttributes.getDimensions();
		new File( outputDirectory ).mkdirs();
		final List< Tuple2< long[], long[] > > intervals = Grids
				.collectAllContainedIntervals( dims, blockSize )
				.stream()
				.map( i -> new Tuple2<>( Intervals.minAsLongArray( i ), Intervals.maxAsLongArray( i ) ) )
				.collect( Collectors.toList() );
		sc
				.parallelize( intervals )
				.mapToPair( minMax -> {
					final long[] blockPos = N5Helpers.blockPos( minMax._1(), blockSize );
					final N5Reader n5reader = N5Helpers.n5Reader( inputN5, N5Helpers.DEFAULT_BLOCK_SIZE );
					final LongArrayDataBlock block = ( LongArrayDataBlock ) n5reader.readBlock( inputDataset, new DatasetAttributes( dims, blockSize, DataType.UINT64, new GzipCompression() ), blockPos );
					return new Tuple2<>( minMax, block.getData() );
				} )
				.flatMapToPair( input -> Arrays
						.stream( input._2() )
						.mapToObj( id -> new Tuple2<>( id, input._1() ) )
						.iterator() )
				.aggregateByKey(
						new ArrayList< Tuple2< long[], long[] > >(),
						( l, v ) -> {
							l.add( v );
							return l;
						},
						( al1, al2 ) -> {
							final ArrayList< Tuple2< long[], long[] > > al = new ArrayList<>();
							al.addAll( al1 );
							al.addAll( al2 );
							return al;
						} )
				.foreach( list -> {
					final String id = list._1().toString();
					// 2 (min, max) * 3 (three dimensions) values of size
					// Long.BYTES per list entry
					final int numBytes = 2 * 3 * list._2().size() * Long.BYTES;
					final byte[] data = new byte[ numBytes ];
					final ByteBuffer bb = ByteBuffer.wrap( data );
					for ( final Tuple2< long[], long[] > t : list._2() )
					{
						final long[] l1 = t._1();
						final long[] l2 = t._2();
						bb.putLong( l1[ 0 ] );
						bb.putLong( l1[ 1 ] );
						bb.putLong( l1[ 2 ] );
						bb.putLong( l2[ 0 ] );
						bb.putLong( l2[ 1 ] );
						bb.putLong( l2[ 2 ] );
					}
					Files.write( data, Paths.get( outputDirectory, id ).toFile() );
				} );
	}

}
