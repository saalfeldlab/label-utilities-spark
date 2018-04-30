package org.janelia.saalfeldlab.multisets.spark.downsample;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.CompressionAdapter;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import picocli.CommandLine;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

public class SparkDownsampler
{

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	private static final String DOWNSAMPLING_FACTORS_KEY = "downsamplingFactors";

	private static final String MAX_NUM_ENTRIES_KEY = "maxNumEntries";

	private static final String MULTI_SCALE_KEY = "multiScale";

	public static class CommandLineParameters implements Callable< Void >
	{

		@Option( names = { "--n5-root", "-r" }, paramLabel = "ROOT", required = true, description = "Input N5 container" )
		private String n5;

		@Option( names = { "--group", "-g" }, paramLabel = "MULTISCALE_GROUP", required = true, description = "Multi scale group(relative to N5). Must contain dataset s0 for first scale level unless --link-mipmap-level-zero is specified." )
		private String multiscaleGroup;

		@Parameters( arity = "1..*", paramLabel = "FACTOR", description = "Factor by which to downscale the input image. Factors are relative to the previous level, not to level zero. Format either fx,fy,fz or f" )
		private String[] factors;

		@Option( names = { "--block-size", "-b" }, paramLabel = "BLOCK_SIZE", description = "Size of the blocks (in cells) to parallelize with Spark. Format either bx,by,bz or b" )
		private String[] blockSize;

		@Option(
				names = { "--max-num-entries", "-m" },
				required = false,
				split = ",",
				description = "Maximum number of multiset entries at each pixels. Values smaller than 1 do not limit the number of entries. Defaults to previous level if not specified. Defaults to -1 for the first level if not specified" )
		private int[] maxNumEntries;

		@Option( names = { "--compression", "-c" }, paramLabel = "COMPRESSION", description = "Compression type to use in output N5 dataset" )
		public String compressionType = "{\"type\":\"gzip\",\"level\":\"-1\"}";

		@Option( names = { "-h", "--help" }, usageHelp = true, description = "display a help message" )
		private boolean helpRequested;

		@Option( names = { "--link-mipmap-level-zero", "-l" }, paramLabel = "PATH_TO_S0", required = false )
		public String pathToMipmapLevelZero;

//		@Option( names = { "--copy-original-dataset", "-C" }, description = "Create a copy of the original data set for the first mipmap level. A symlink is created by default" )
//		private boolean copyDataset;

		public int[][] getFactors()
		{
			return Arrays.stream( factors ).map( SparkDownsampler::toIntegerArray ).toArray( int[][]::new );
		}

		public int[][] getBlockSizes()
		{
			return Arrays.stream( blockSize ).map( SparkDownsampler::toIntegerArray ).toArray( int[][]::new );
		}

		@Override
		public Void call() throws Exception
		{
			LOG.warn( "Will downsample with these factors: {}", Arrays.toString( factors ) );
			blockSize = blockSize == null ? new String[ 0 ] : blockSize;
			maxNumEntries = maxNumEntries == null ? new int[] { -1 } : maxNumEntries;
			final String defaultBlockSize = Arrays.stream( blockSize ).limit( factors.length ).reduce( ( f, s ) -> s ).orElse( "64" );
			blockSize = Stream.concat(
					Arrays.stream( blockSize ).limit( factors.length ),
					Stream.generate( () -> defaultBlockSize ).limit( factors.length - blockSize.length ) ).toArray( String[]::new );
			LOG.debug( "Block sizes: {}", Arrays.toString( blockSize ) );
			final int[][] factors = getFactors();
			final int[][] blockSizes = getBlockSizes();
			final Compression compression = fromString( compressionType );

			for ( int i = 0; i < factors.length; ++i )
				if ( !checkScaleFactors( factors[ i ] ) )
				{
					LOG.error( "Got illegal downscaling factors: {}", factors[ i ] );
					throw new IllegalArgumentException( "Got illegal downscaling factors: " + Arrays.toString( factors[ i ] ) );
				}

			addMultiScaleTag( new N5FSWriter( n5 ), multiscaleGroup );

			final SparkConf conf = new SparkConf().setAppName( "SparkDownsampler" );
			try (final JavaSparkContext sc = new JavaSparkContext( conf ))
			{

				if ( pathToMipmapLevelZero != null )
				{
					final Path linkLocation = Paths.get( n5, multiscaleGroup, "s0" );
					linkLocation.getParent().toFile().mkdirs();
					LOG.debug( "Creating link at `{}' pointing to `{}'", linkLocation, pathToMipmapLevelZero );
					try
					{
						Files.createSymbolicLink( linkLocation, Paths.get( pathToMipmapLevelZero ) );
					}
					catch ( final FileAlreadyExistsException e )
					{
						LOG.warn( "Dataset for mipmap level already exists at {}, will not create symlink!", linkLocation );
					}
				}
				int lastMaxNumEntries = -1;
				for ( int factorIndex = 0, level = 1; factorIndex < factors.length; ++factorIndex, ++level )
				{
					final String previousScaleLevel = multiscaleGroup + "/s" + ( level - 1 );
					final String currentScaleLevel = multiscaleGroup + "/s" + level;

					final int maxNumEntries = factorIndex < this.maxNumEntries.length ? this.maxNumEntries[ factorIndex ] : lastMaxNumEntries;
					lastMaxNumEntries = maxNumEntries;

					SparkDownsampler.downsample( sc,
							new N5FSReader( n5 ),
							n5,
							previousScaleLevel,
							factors[ factorIndex ],
							blockSizes[ factorIndex ],
							n5,
							currentScaleLevel,
							compression,
							maxNumEntries );
				}
			}

			return null;
		}

	}

	public static void main( final String[] args ) throws IOException
	{
		run( args );
	}

	public static void run( final String[] args ) throws IOException
	{
		CommandLine.call( new CommandLineParameters(), System.err, args );
	}

	public static void downsample( final JavaSparkContext sc,
			final N5Reader reader,
			final String readGroupName,
			final String readDatasetName,
			final int[] downsampleFactor,
			final int[] blockSize,
			final String outputGroupName,
			final String outputDatasetName,
			final Compression compression,
			final int maxNumEntries ) throws IOException
	{

		final DatasetAttributes attributes = reader.getDatasetAttributes( readDatasetName );

		if ( !checkBlockSize( attributes.getBlockSize(), blockSize, downsampleFactor ) )
		{
			LOG.error( "Got illegal block sizefactors: previous={}, current={}, factors={}", attributes.getBlockSize(), blockSize, downsampleFactor );
			throw new IllegalArgumentException( "Got illegal downscaling factors:" );
		}

		final long[] dimensions = attributes.getDimensions();
		final long[] max = Arrays.stream( dimensions ).map( dim -> dim - 1 ).toArray();

		final List< long[] > positions = new ArrayList<>();

		final int nDim = attributes.getNumDimensions();

		final long[] downsampledDimensions = new long[ nDim ];
		// needs to be cast to long, not int
		Arrays.setAll( downsampledDimensions, i -> ( long ) Math.ceil( ( double ) dimensions[ i ] / downsampleFactor[ i ] ) );

		final long[] offset = new long[ nDim ];

		for ( int d = 0; d < nDim; )
		{

			positions.add( offset.clone() );

			for ( d = 0; d < nDim; d++ )
			{
				offset[ d ] += blockSize[ d ];
				if ( offset[ d ] < downsampledDimensions[ d ] )
					break;
				else
					offset[ d ] = 0;
			}
		}

		final N5Writer writer = new N5FSWriter( outputGroupName );
		final double[] previousDownsamplingFactor = Optional
				.ofNullable( reader.getAttribute( readDatasetName, DOWNSAMPLING_FACTORS_KEY, double[].class ) )
				.orElse( DoubleStream.generate( () -> 1.0 ).limit( nDim ).toArray() );
		final double[] accumulatedDownsamplingFactor = IntStream.range( 0, nDim ).mapToDouble( d -> previousDownsamplingFactor[ d ] * downsampleFactor[ d ] ).toArray();
		writer.createDataset( outputDatasetName, downsampledDimensions, blockSize, DataType.UINT8, compression );
		writer.setAttribute( outputDatasetName, DOWNSAMPLING_FACTORS_KEY, accumulatedDownsamplingFactor );
		writer.setAttribute( outputDatasetName, MAX_NUM_ENTRIES_KEY, maxNumEntries );

		sc.parallelize( positions )
				.map( new MinToInterval( max, blockSize ) )
				.foreach( new SparkDownsampleFunction( readGroupName, readDatasetName, downsampleFactor, outputGroupName, outputDatasetName, maxNumEntries ) );

//		System.out.println( "Across " + positions.size() + " parallelized sections, " + output + " cells were downscaled" );
	}

	public static int[] toIntegerArray( final String str )
	{
		return toIntegerArray( str, 3, "," );
	}

	public static int[] toIntegerArray( final String str, final int requiredNumberOfFields, final String splitRegex )
	{
		return toIntegerArray( str, new int[ requiredNumberOfFields ], splitRegex );
	}

	public static int[] toIntegerArray( final String str, final int[] target, final String splitRegex )
	{
		final String[] split = str.split( splitRegex );
		for ( int i = 0; i < split.length; ++i )
			target[ i ] = Integer.parseInt( split[ i ] );
		for ( int k = split.length; k < target.length; ++k )
			target[ k ] = Integer.parseInt( split[ split.length - 1 ] );

		return target;
	}

	public static Compression fromString( final String str )
	{
		final GsonBuilder gsonBuilder = new GsonBuilder();
		gsonBuilder.registerTypeHierarchyAdapter( Compression.class, CompressionAdapter.getJsonAdapter() );
		final Gson gson = gsonBuilder.create();
		return gson.fromJson( str, Compression.class );
	}

	public static boolean checkScaleFactors( final int[] scaleFactors )
	{
		for ( final int factor : scaleFactors )
			if ( factor < 1 )
				return false;
		return true;
	}

	/**
	 *
	 * Each block at lower resolution must border-align with the scaled blocks
	 * at the lower resolution, i.e. the block size of a lower dimension block
	 * multiplied with the scale facotr must be an integer multiple of the block
	 * size at higher resolution for all dimensions.
	 *
	 * @param blockSizePrevious
	 * @param blockSizeCurrent
	 * @param scaleFactors
	 * @return
	 */
	public static boolean checkBlockSize(
			final int[] blockSizePrevious,
			final int[] blockSizeCurrent,
			final int[] scaleFactors )
	{
		for ( int d = 0; d < scaleFactors.length; ++d )
			if ( blockSizeCurrent[ d ] * scaleFactors[ d ] % blockSizePrevious[ d ] != 0 )
				return false;
		return true;
	}

	public static void addMultiScaleTag( final N5Writer n5, final String group ) throws IOException
	{
		if ( !n5.exists( group ) )
			n5.createGroup( group );
		n5.setAttribute( group, MULTI_SCALE_KEY, true );
	}
}
