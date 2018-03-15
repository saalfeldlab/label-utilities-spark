package bdv.bigcat.spark;

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

import gnu.trove.list.array.TIntArrayList;
import picocli.CommandLine;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

public class SparkDownsampler
{

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	private static final String DOWNSAMPLING_FACTORS_KEY = "downsamplingFactors";

	public static class CommandLineParameters implements Callable< Void >
	{

		@Option( names = { "--group", "-g" }, required = true, description = "Input group name (N5 group)" )
		private String group;

		@Option( names = { "--dataset", "-d" }, required = true, description = "Input dataset name (N5 relative path from group)" )
		private String dataset;

		@Parameters( arity = "1..*", description = "Factor by which to downscale the input image. Factors are relative to the previous level, not to level zero. Format either fx,fy,fz or f" )
		private String[] factors;

		@Option( names = { "--block-size", "-b" }, description = "Size of the blocks (in cells) to parallelize with Spark. Format either bx,by,bz or b" )
		private String[] blockSize;

		@Option(
				names = { "--max-num-entries", "-m" },
				required = false,
				split = ",",
				description = "Maximum number of multiset entries at each pixels. Values smaller than 1 do not limit the number of entries. Defaults to previous level if not specified. Defaults to -1 for the first level if not specified" )
		private int[] maxNumEntries;

		@Option( names = { "--compression", "-c" }, description = "Compression type to use in output N5 dataset" )
		public String compressionType = "{\"type\":\"raw\"}";

		@Option( names = { "-h", "--help" }, usageHelp = true, description = "display a help message" )
		private boolean helpRequested;

		@Option( names = { "--link-mipmap-level-zero", "-l" }, required = false )
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

			final SparkConf conf = new SparkConf().setAppName( "SparkDownsampler" );
			final TIntArrayList maxNumEntriesList = new TIntArrayList();
			try (final JavaSparkContext sc = new JavaSparkContext( conf ))
			{

				if ( pathToMipmapLevelZero != null )
				{
					final Path linkLocation = Paths.get( group, dataset, "s0" );
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
					final String previousScaleLevel = dataset + "/s" + ( level - 1 );
					final String currentScaleLevel = dataset + "/s" + level;

					final int maxNumEntries = factorIndex < this.maxNumEntries.length ? this.maxNumEntries[ factorIndex ] : lastMaxNumEntries;
					lastMaxNumEntries = maxNumEntries;
					maxNumEntriesList.add( maxNumEntries );

					SparkDownsampler.downsample( sc,
							new N5FSReader( group ),
							group,
							previousScaleLevel,
							factors[ factorIndex ],
							blockSizes[ factorIndex ],
							group,
							currentScaleLevel,
							compression,
							maxNumEntries );
				}
			}

			System.out.println( "YO " + maxNumEntriesList );
			return null;
		}

	}

	public static void main( final String[] args ) throws IOException
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
}
