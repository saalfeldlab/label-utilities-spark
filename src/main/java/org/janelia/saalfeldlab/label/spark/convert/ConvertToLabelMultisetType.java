package org.janelia.saalfeldlab.label.spark.convert;

import java.io.IOException;
import java.io.Serializable;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang.time.DurationFormatUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.label.spark.N5Helpers;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.CompressionAdapter;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5LabelMultisets;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.converter.Converters;
import net.imglib2.type.NativeType;
import net.imglib2.type.label.FromIntegerTypeConverter;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import picocli.CommandLine;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import scala.Tuple2;

public class ConvertToLabelMultisetType
{

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static final String LABEL_MULTISETTYPE_KEY = "isLabelMultiset";

	public static final String DATA_TYPE_KEY = "dataType";

	public static final String COMPRESSION_KEY = "compression";

	public static final String BLOCK_SIZE_KEY = "blockSize";

	public static final String DIMENSIONS_KEY = "dimensions";

	public static final String MAX_ID_KEY = N5Helpers.MAX_ID_KEY;

	public static final int DEFAULT_BLOCK_SIZE = 64;

	public static final int MAX_PARTITIONS = 1000;

	// TODO make this parallizable/spark and not hdf-to-n5 but convert between
	// various instances of n5 instead
	static public class CommandLineParameters implements Callable< Void >
	{
		@Option( names = { "--input-n5", "-i" }, paramLabel = "INPUT_N5", required = true, description = "Input N5 container. Currently supports N5 and HDF5." )
		private String inputN5;

		@Option( names = { "--output-n5", "-o" }, paramLabel = "OUTPUT_N5", description = "Output N5 container. Defaults to INPUT_N5" )
		private String outputN5;

		@Option( names = { "--dataset", "-d" }, paramLabel = "INPUT_DATASET", required = true, description = "Input dataset name (relative to INPUT_N5" )
		private String inputDataset;

		@Parameters( arity = "1", paramLabel = "OUTPUT_DATASET", description = "Output dataset name (relative to OUTPUT_N5)" )
		private String outputDatasetName;

		@Option( names = { "--block-size", "-b" }, paramLabel = "BLOCK_SIZE", description = "Size of cells to use in the output N5 dataset. Defaults to 64. Either single integer value for isotropic block size or comma-seperated list of block size per dimension", split = "," )
		private int[] blockSize;

		@Option( names = { "--compression", "-c" }, paramLabel = "COMPRESSION", description = "Compression type to use in output N5 dataset" )
		public String compressionType = "{\"type\":\"gzip\",\"level\":-1}";

		@Option(
				names = { "--revert-array-attributes" },
				required = false,
				description = "When copying, revert all additional array attributes that are not dataset attributes. E.g. [3,2,1] -> [1,2,3]" )
		private boolean revertArrayAttributes;

		@Override
		public Void call() throws IOException
		{
			this.blockSize = this.blockSize == null || this.blockSize.length == 0 ? new int[] { DEFAULT_BLOCK_SIZE } : this.blockSize;
			this.outputN5 = this.outputN5 == null ? this.inputN5 : this.outputN5;

			final Gson gson = new GsonBuilder()
					.registerTypeHierarchyAdapter( Compression.class, CompressionAdapter.getJsonAdapter() )
					.create();
			final Compression compression = new GzipCompression();// .fromJson(
																	// compressionType,
																	// Compression.class
																	// );
			final int nDim = N5Helpers.n5Reader( this.inputN5 ).getDatasetAttributes( this.inputDataset ).getNumDimensions();
			final int[] blockSize = this.blockSize.length < nDim ? IntStream.generate( () -> this.blockSize[ 0 ] ).limit( nDim ).toArray() : this.blockSize;

			final long startTime = System.currentTimeMillis();

			final SparkConf conf = new SparkConf().setAppName( MethodHandles.lookup().lookupClass().getName() );

			try (final JavaSparkContext sc = new JavaSparkContext( conf ))
			{
				convertToLabelMultisetType(
						sc,
						inputN5,
						inputDataset,
						blockSize,
						outputN5,
						outputDatasetName,
						compression,
						revertArrayAttributes );
			}

			final long endTime = System.currentTimeMillis();

			final String formattedTime = DurationFormatUtils.formatDuration( endTime - startTime, "HH:mm:ss.SSS" );
			System.out.println( "Converted " + inputN5 + " to N5 dataset at " + outputN5 + " with name " + outputDatasetName +
					" in " + formattedTime );
			return null;

		}

	}

	public static void main( final String... args ) throws IOException
	{
		run( args );
	}

	public static void run( final String... args ) throws IOException
	{
		System.out.println( "Command line arguments: " + Arrays.toString( args ) );
		LOG.debug( "Command line arguments: ", Arrays.toString( args ) );
		CommandLine.call( new CommandLineParameters(), System.err, args );

	}

	public static < I extends IntegerType< I > & NativeType< I > > void convertToLabelMultisetType(
			final JavaSparkContext sc,
			final String inputGroup,
			final String inputDataset,
			final int[] blockSize,
			final String outputGroupName,
			final String outputDatasetName,
			final Compression compression,
			final boolean revert ) throws IOException
	{
		final N5Reader reader = N5Helpers.n5Reader( inputGroup, blockSize );
		final int[] inputBlockSize = reader.getDatasetAttributes( inputDataset ).getBlockSize();
		final RandomAccessibleInterval< I > img = N5Utils.open( reader, inputDataset );
		final Map< String, Class< ? > > attributeNames = reader.listAttributes( inputDataset );
		Arrays.asList(
				LABEL_MULTISETTYPE_KEY,
				DATA_TYPE_KEY,
				COMPRESSION_KEY,
				BLOCK_SIZE_KEY,
				DIMENSIONS_KEY )
				.forEach( attributeNames::remove );

		final int nDim = img.numDimensions();

		final long[] dimensions = new long[ nDim ];
		img.dimensions( dimensions );

		final N5Writer writer = N5Helpers.n5Writer( outputGroupName, blockSize );
		writer.createDataset( outputDatasetName, dimensions, blockSize, DataType.UINT8, compression );
		writer.setAttribute( outputDatasetName, LABEL_MULTISETTYPE_KEY, true );
		for ( final Entry< String, Class< ? > > entry : attributeNames.entrySet() )
			writer.setAttribute( outputDatasetName, entry.getKey(), N5Helpers.revertInplaceAndReturn( reader.getAttribute( inputDataset, entry.getKey(), entry.getValue() ), revert ) );

		final int[] parallelizeBlockSize = new int[ blockSize.length ];
		if ( Intervals.numElements( blockSize ) >= Intervals.numElements( inputBlockSize ) )
		{
			Arrays.setAll( parallelizeBlockSize, d -> blockSize[ d ] );
			LOG.debug( "Output block size {} is the same or bigger than the input block size {}, parallelizing over output blocks of size {}", blockSize, inputBlockSize, parallelizeBlockSize );
		}
		else
		{
			Arrays.setAll( parallelizeBlockSize, d -> ( int ) Math.max( Math.round( ( double ) inputBlockSize[ d ] / blockSize[ d ] ), 1 ) * blockSize[ d ] );
			LOG.debug( "Output block size {} is smaller than the input block size {}, parallelizing over adjusted input blocks of size {}", blockSize, inputBlockSize, parallelizeBlockSize );
		}

		final List< Tuple2< long[], long[] > > intervals = Grids.collectAllContainedIntervals( dimensions, parallelizeBlockSize )
				.stream()
				.map( interval -> new Tuple2<>( Intervals.minAsLongArray( interval ), Intervals.maxAsLongArray( interval ) ) )
				.collect( Collectors.toList() );

		final long maxId = sc
				.parallelize( intervals, Math.min( intervals.size(), MAX_PARTITIONS ) )
				.map( intervalMinMax -> {
					final Interval interval = new FinalInterval( intervalMinMax._1(), intervalMinMax._2() );

					@SuppressWarnings("unchecked")
					final RandomAccessibleInterval< I > blockImg = Views.interval(
							( RandomAccessibleInterval< I > ) N5Utils.open( N5Helpers.n5Reader( inputGroup, blockSize ), inputDataset ),
							interval
						);

					final FromIntegerTypeConverter< I > converter = new FromIntegerTypeConverter<>();
					final LabelMultisetType type = FromIntegerTypeConverter.getAppropriateType();
					long blockMaxId = Long.MIN_VALUE;
					for ( final I i : Views.iterable( blockImg ) )
					{
						final long il = i.getIntegerLong();
						blockMaxId = Math.max( il, blockMaxId );
					}
					final RandomAccessibleInterval< LabelMultisetType > converted = Converters.convert( blockImg, converter, type );

					N5LabelMultisets.saveLabelMultisetBlock(
							converted,
							N5Helpers.n5Writer( outputGroupName, blockSize ),
							outputDatasetName,
							computeGridOffset( interval, blockSize ) // TODO: this parameter can be omitted with next release of n5-imglib2
						);

					return blockMaxId;
				} )
				.max( new LongComparator() );

		writer.setAttribute( outputDatasetName, MAX_ID_KEY, maxId );
	}

	private static long[] computeGridOffset( final Interval source, final int[] blockSize )
	{
		final long[] gridOffset = new long[blockSize.length];
		Arrays.setAll(gridOffset, d -> source.min(d) / blockSize[d]);
		return gridOffset;
	}

	private static final class LongComparator implements Comparator< Long >, Serializable
	{
		@Override
		public int compare( final Long o1, final Long o2 )
		{
			return Long.compare( o1, o2 );
		}
	}
}
