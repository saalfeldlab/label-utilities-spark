package org.janelia.saalfeldlab.multisets.spark.uniquelabels;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.lang.time.DurationFormatUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.multisets.spark.exception.InputSameAsOutput;
import org.janelia.saalfeldlab.multisets.spark.exception.InvalidDataType;
import org.janelia.saalfeldlab.multisets.spark.exception.InvalidDataset;
import org.janelia.saalfeldlab.multisets.spark.exception.InvalidN5Container;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Writer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.imglib2.algorithm.util.Grids;
import net.imglib2.util.Intervals;
import picocli.CommandLine;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import scala.Tuple2;

public class ExtractUniqueLabelsPerBlock
{

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static final String LABEL_MULTISETTYPE_KEY = "isLabelMultiset";

	public static final int DEFAULT_BLOCK_SIZE = 64;

	// TODO make this parallizable/spark and not hdf-to-n5 but convert between
	// various instances of n5 instead
	static public class CommandLineParameters implements Callable< Void >
	{

		@Parameters( index = "0", paramLabel = "INPUT_N5", description = "Input N5 container" )
		private String inputN5;

		@Parameters( index = "1", paramLabel = "INPUT_DATASET", description = "Input dataset. Must be LabelMultisetType or integer type" )
		private String inputDataset;

		@Option( names = { "--output-n5", "-o" }, paramLabel = "OUTPUT_N5", description = "Output N5 container. Defaults to INPUT_N5" )
		private String outputN5;

		@Parameters( index = "2", paramLabel = "OUTPUT_DATASET", description = "Output dataset name (relative to OUTPUT_N5)" )
		private String outputDatasetName;

		@Override
		public Void call() throws Exception
		{
			this.outputN5 = this.outputN5 == null ? this.inputN5 : this.outputN5;

			if ( this.outputN5 == null ) { throw new InvalidN5Container( "OUTPUT_N5", this.outputN5 ); }

			if ( this.inputN5 == null ) { throw new InvalidN5Container( "INPUT_N5", this.inputN5 ); }

			if ( this.inputDataset == null ) { throw new InvalidDataset( "INPUT_DATASET", inputDataset ); }

			if ( this.outputDatasetName == null ) { throw new InvalidDataset( "OUTPUT_DATASET", outputDatasetName ); }

			if ( this.inputN5.equals( this.outputN5 ) && this.inputDataset.equals( this.outputDatasetName ) ) { throw new InputSameAsOutput( this.inputN5, this.inputDataset ); }

			final N5Reader reader = n5Reader( this.inputN5 );
			final DatasetAttributes inputAttributes = reader.getDatasetAttributes( this.inputDataset );
			final int[] blockSize = inputAttributes.getBlockSize();
			final long[] dims = inputAttributes.getDimensions();

			final boolean isMultisetType = Optional
					.ofNullable( reader.getAttribute( this.inputDataset, LABEL_MULTISETTYPE_KEY, Boolean.class ) )
					.orElse( false );

			if ( !isMultisetType )
			{
				final DataType dataType = inputAttributes.getDataType();
				switch ( dataType )
				{
				case FLOAT32:
				case FLOAT64:
					throw new InvalidDataType( dataType );
				default:
					break;
				}
			}

			final long startTime = System.currentTimeMillis();

			final SparkConf conf = new SparkConf().setAppName( MethodHandles.lookup().lookupClass().getName() );

			final N5Writer writer = n5Writer( this.outputN5, blockSize );
			// TODO should we actually compress at all? unique lables might not
			// compress very well
			final DatasetAttributes outputAttributes = new DatasetAttributes( dims, blockSize, DataType.UINT64, new GzipCompression() );
			writer.createDataset( this.outputDatasetName, outputAttributes );

			try (final JavaSparkContext sc = new JavaSparkContext( conf ))
			{
				final List< Tuple2< long[], long[] > > intervals = Grids
						.collectAllContainedIntervals( dims, blockSize )
						.stream()
						.map( i -> new Tuple2<>( Intervals.minAsLongArray( i ), Intervals.maxAsLongArray( i ) ) )
						.collect( Collectors.toList() );
				sc
						.parallelize( intervals )
						.foreach(
								new ExtractAndStoreLabelList(
										inputN5,
										outputN5,
										inputDataset,
										outputDatasetName,
										dims,
										blockSize,
										isMultisetType ) );
			}

			final long endTime = System.currentTimeMillis();

			final String formattedTime = DurationFormatUtils.formatDuration( endTime - startTime, "HH:mm:ss.SSS" );
			System.out.println( "Converted " + inputN5 + " to N5 dataset at " + outputN5 + " with name " + outputDatasetName +
					" in " + formattedTime );
			return null;

		}

	}

	public static void run( final String... args ) throws IOException
	{
		System.out.println( "Command line arguments: " + Arrays.toString( args ) );
		LOG.debug( "Command line arguments: {}", Arrays.toString( args ) );
		CommandLine.call( new CommandLineParameters(), System.err, args );

	}

	public static N5Reader n5Reader( final String base, final int... defaultCellDimensions ) throws IOException
	{
		return isHDF( base ) ? new N5HDF5Reader( base, defaultCellDimensions ) : new N5FSReader( base );
	}

	public static N5Writer n5Writer( final String base, final int... defaultCellDimensions ) throws IOException
	{
		return isHDF( base ) ? new N5HDF5Writer( base, defaultCellDimensions ) : new N5FSWriter( base );
	}

	public static boolean isHDF( final String base )
	{
		LOG.debug( "Checking {} for HDF", base );
		final boolean isHDF = Pattern.matches( "^h5://", base ) || Pattern.matches( "^.*\\.(hdf|h5)$", base );
		LOG.debug( "{} is hdf5? {}", base, isHDF );
		return isHDF;
	}

	public static long[] blockPos( final long[] position, final int[] blockSize )
	{
		final long[] blockPos = new long[ position.length ];
		Arrays.setAll( blockPos, d -> position[ d ] / blockSize[ d ] );
		return blockPos;
	}
}
