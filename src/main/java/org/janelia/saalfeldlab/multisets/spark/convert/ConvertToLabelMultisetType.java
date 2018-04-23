package org.janelia.saalfeldlab.multisets.spark.convert;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import org.apache.commons.lang.time.DurationFormatUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.CompressionAdapter;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.label.LabelUtils;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import picocli.CommandLine;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

public class ConvertToLabelMultisetType
{

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static final String LABEL_MULTISETTYPE_KEY = "isLabelMultiset";

	public static final String DATA_TYPE_KEY = "dataType";

	public static final String COMPRESSION_KEY = "compression";

	public static final String BLOCK_SIZE_KEY = "blockSize";

	public static final String DIMENSIONS_KEY = "dimensions";

	public static final int DEFAULT_BLOCK_SIZE = 64;

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
			final int nDim = n5Reader( this.inputN5 ).getDatasetAttributes( this.inputDataset ).getNumDimensions();
			final int[] blockSize = this.blockSize.length < nDim ? IntStream.generate( () -> this.blockSize[ 0 ] ).limit( nDim ).toArray() : this.blockSize;

			final long startTime = System.currentTimeMillis();

			final SparkConf conf = new SparkConf().setAppName( MethodHandles.lookup().lookupClass().getName() );

			try (final JavaSparkContext sc = new JavaSparkContext( conf ))
			{
				convertHDF5toN5(
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
		System.out.println( "Command line arguments: " + Arrays.toString( args ) );
		LOG.debug( "Command line arguments: ", Arrays.toString( args ) );
		CommandLine.call( new CommandLineParameters(), System.err, args );

	}

	public static < I extends IntegerType< I > & NativeType< I > > void convertHDF5toN5(
			final JavaSparkContext sc,
			final String inputGroup,
			final String inputDataset,
			final int[] blockSize,
			final String outputGroupName,
			final String outputDatasetName,
			final Compression compression,
			final boolean revert ) throws IOException
	{
		final N5Reader reader = n5Reader( inputGroup, blockSize );
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

		final N5Writer writer = n5Writer( outputGroupName, blockSize );
		writer.createDataset( outputDatasetName, dimensions, blockSize, DataType.UINT8, compression );
		writer.setAttribute( outputDatasetName, LABEL_MULTISETTYPE_KEY, true );
		for ( final Entry< String, Class< ? > > entry : attributeNames.entrySet() )
			writer.setAttribute( outputDatasetName, entry.getKey(), revertInplaceAndReturn( reader.getAttribute( inputDataset, entry.getKey(), entry.getValue() ), revert ) );

		final List< long[] > offsets = new ArrayList<>();

		final long[] min = new long[ nDim ];

		for ( int dim = 0; dim < nDim; )
		{

			offsets.add( min.clone() );

			for ( dim = 0; dim < nDim; dim++ )
			{
				min[ dim ] += blockSize[ dim ];
				if ( min[ dim ] < dimensions[ dim ] )
					break;
				else
					min[ dim ] = 0;
			}
		}

		// required to serialize compression as json string
		final Gson gson = new GsonBuilder()
				.registerTypeHierarchyAdapter( Compression.class, CompressionAdapter.getJsonAdapter() )
				.create();

		sc
				.parallelize( offsets )
				.map( new ToInterval( dimensions, blockSize ) )
				.map( new ReadIntegerData< I >( inputGroup, inputDataset, blockSize ) )
				.map( new ConvertToLabelMultisetTypeFunction<>() )
				.mapToPair( new AttachBlockPosition<>( blockSize ) )
				.map( new ConvertToDataBlock() )
				.foreach( new WriteBlock<>( outputGroupName, outputDatasetName, dimensions, blockSize, DataType.UINT8, gson.toJson( compression ) ) );

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

	public static ByteArrayDataBlock toDataBlock(
			final RandomAccessibleInterval< LabelMultisetType > multisets,
			final long[] cellOffset )
	{
		return new ByteArrayDataBlock(
				Intervals.dimensionsAsIntArray( multisets ),
				cellOffset,
				LabelUtils.serializeLabelMultisetTypes( Views.flatIterable( multisets ),
						( int ) Intervals.numElements( multisets ) ) );
	}

	public static long[] blockPos( final long[] position, final int[] blockSize )
	{
		final long[] blockPos = new long[ position.length ];
		Arrays.setAll( blockPos, d -> position[ d ] / blockSize[ d ] );
		return blockPos;
	}

	public static < T > T revertInplaceAndReturn( final T t, final boolean revert )
	{
		if ( !revert )
			return t;

		if ( t instanceof boolean[] )
		{
			final boolean[] arr = ( boolean[] ) t;
			for ( int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k )
			{
				final boolean v = arr[ 0 ];
				arr[ 0 ] = arr[ k ];
				arr[ k ] = v;
			}
		}

		if ( t instanceof byte[] )
		{
			final byte[] arr = ( byte[] ) t;
			for ( int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k )
			{
				final byte v = arr[ 0 ];
				arr[ 0 ] = arr[ k ];
				arr[ k ] = v;
			}
		}

		if ( t instanceof char[] )
		{
			final char[] arr = ( char[] ) t;
			for ( int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k )
			{
				final char v = arr[ 0 ];
				arr[ 0 ] = arr[ k ];
				arr[ k ] = v;
			}
		}

		if ( t instanceof short[] )
		{
			final short[] arr = ( short[] ) t;
			for ( int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k )
			{
				final short v = arr[ 0 ];
				arr[ 0 ] = arr[ k ];
				arr[ k ] = v;
			}
		}

		if ( t instanceof int[] )
		{
			final int[] arr = ( int[] ) t;
			for ( int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k )
			{
				final int v = arr[ 0 ];
				arr[ 0 ] = arr[ k ];
				arr[ k ] = v;
			}
		}

		if ( t instanceof long[] )
		{
			final long[] arr = ( long[] ) t;
			for ( int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k )
			{
				final long v = arr[ 0 ];
				arr[ 0 ] = arr[ k ];
				arr[ k ] = v;
			}
		}

		if ( t instanceof float[] )
		{
			final float[] arr = ( float[] ) t;
			for ( int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k )
			{
				final float v = arr[ 0 ];
				arr[ 0 ] = arr[ k ];
				arr[ k ] = v;
			}
		}

		if ( t instanceof double[] )
		{
			final double[] arr = ( double[] ) t;
			for ( int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k )
			{
				final double v = arr[ 0 ];
				arr[ 0 ] = arr[ k ];
				arr[ k ] = v;
			}
		}

		return t;
	}
}
