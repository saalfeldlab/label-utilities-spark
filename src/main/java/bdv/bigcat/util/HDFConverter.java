package bdv.bigcat.util;

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

public class HDFConverter
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
		@Option( names = { "--input-group", "-g" }, required = true, description = "Input N5 group." )
		private String inputGroup;

		@Option( names = { "--output-group", "-G" }, description = "Output group name (N5 group). Defaults to INPUT_GROUP" )
		private String outputGroupName;

		@Option( names = { "--dataset", "-d" }, required = true, description = "Input dataset name (N5 relative path from group)" )
		private String inputDataset;

		@Parameters( arity = "1", description = "Output dataset relative to OUTPUT_GROUP" )
		private String outputDatasetName;

		@Option( names = { "--block-size", "-b" }, description = "Size of cells to use in the output N5 dataset. Defaults to 64", split = "," )
		private int[] blockSize;

		@Option( names = { "--compression", "-c" }, description = "Compression type to use in output N5 dataset" )
		public String compressionType = "{\"type\":\"gzip\",\"level\":-1}";

		@Override
		public Void call() throws IOException
		{
			this.blockSize = this.blockSize == null || this.blockSize.length == 0 ? new int[] { DEFAULT_BLOCK_SIZE } : this.blockSize;

			final Gson gson = new GsonBuilder()
					.registerTypeHierarchyAdapter( Compression.class, CompressionAdapter.getJsonAdapter() )
					.create();
			final Compression compression = new GzipCompression();// .fromJson(
																	// compressionType,
																	// Compression.class
																	// );
			final int nDim = n5Reader( this.inputGroup ).getDatasetAttributes( this.inputDataset ).getNumDimensions();
			final int[] blockSize = this.blockSize.length < nDim ? IntStream.generate( () -> this.blockSize[ 0 ] ).limit( nDim ).toArray() : this.blockSize;

			final long startTime = System.currentTimeMillis();

			final SparkConf conf = new SparkConf().setAppName( MethodHandles.lookup().lookupClass().getName() );

			try (final JavaSparkContext sc = new JavaSparkContext( conf ))
			{
				convertHDF5toN5(
						sc,
						inputGroup,
						inputDataset,
						blockSize,
						outputGroupName,
						outputDatasetName,
						compression );
			}

			final long endTime = System.currentTimeMillis();

			final String formattedTime = DurationFormatUtils.formatDuration( endTime - startTime, "HH:mm:ss.SSS" );
			System.out.println( "Converted " + inputGroup + " to N5 dataset at " + outputGroupName + " with name " + outputDatasetName +
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
			final Compression compression ) throws IOException
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
			writer.setAttribute( outputDatasetName, entry.getKey(), reader.getAttribute( inputDataset, entry.getKey(), entry.getValue() ) );

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
				.map( new ConvertToLabelMultisetType<>() )
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
}
