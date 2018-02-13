package bdv.bigcat.spark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.google.gson.Gson;

public class SparkDownsampler
{
	static public class Parameters
	{

		@Parameter( names = { "--igroupname", "--igroup", "-ig" }, description = "Input group name (N5 group)" )
		public String inputGroupName = null;

		@Parameter( names = { "--idatasetname", "--idata", "-id" }, description = "Input dataset name (N5 relative path from group)" )
		public String inputDatasetName = null;

		@Parameter( names = { "--ogroupname", "--ogroup", "-og" }, description = "Output group name (N5 group)" )
		public String outputGroupName = null;

		@Parameter( names = { "--odatasetname", "--odata", "-od" }, description = "Output dataset name (N5 relative path from group)" )
		public String outputDatasetName = null;

		@Parameter( names = { "--factor", "-f" }, description = "Factor by which to downscale the input image" )
		public List< Integer > factor = new ArrayList<>();

		@Parameter( names = { "--parallelblocks", "-pb" }, description = "Size of the blocks (in cells) to parallelize with Spark" )
		public List< Integer > parallelBlockSize = new ArrayList<>();

		@Parameter( names = { "--compression", "-c" }, description = "Compression type to use in output N5 dataset" )
		public String compressionType = "RAW";

		public boolean init()
		{
			if ( inputGroupName == null || inputDatasetName == null || factor.size() == 0 )
				return false;

			if ( parallelBlockSize.size() == 0 )
				for ( int i = 0; i < factor.size(); i++ )
					parallelBlockSize.add( 16 );

			if ( outputGroupName == null )
				outputGroupName = inputGroupName; // default to using same group

			if ( outputDatasetName == null )
			{
				outputDatasetName = inputDatasetName + "-downscaled-";
				for ( int i = 0; i < factor.size(); i++ )
					outputDatasetName += ( i != 0 ? "x" : "" ) + i;
			}
			return true;
		}
	}

	public static void main( final String[] args ) throws IOException
	{
		final Parameters params = new Parameters();
		final JCommander commander = new JCommander( params, args );

		if ( !params.init() )
		{
			commander.usage();
			return;
		}

		final SparkConf conf = new SparkConf().setAppName( "SparkDownsampler" );
		final JavaSparkContext sc = new JavaSparkContext( conf );
		final N5FSReader reader = new N5FSReader( params.inputGroupName );
		final Gson gson = reader.getGson();
		SparkDownsampler.downsample( sc,
				reader,
				params.inputGroupName,
				params.inputDatasetName,
				params.factor.stream().mapToInt( i -> i ).toArray(), params.parallelBlockSize.stream().mapToInt( i -> i ).toArray(),
				params.outputGroupName, params.outputDatasetName,
				gson.fromJson( params.compressionType, Compression.class ) );
	}

	public static void downsample( final JavaSparkContext sc,
			final N5Reader reader, final String readGroupName, final String readDatasetName,
			final int[] downsampleFactor, final int[] parallelSize,
			final String outputGroupName, final String outputDatasetName,
			final Compression compression ) throws IOException
	{

		final DatasetAttributes attributes = reader.getDatasetAttributes( readDatasetName );

		final long[] dimensions = attributes.getDimensions();
		final int[] blockSize = attributes.getBlockSize();

		final List< DownsampleBlock > parallelizeSections = new ArrayList<>();

		final int nDim = attributes.getNumDimensions();

		final long[] downsampledDimensions = new long[ nDim ];
		Arrays.setAll( downsampledDimensions, i -> ( int ) Math.ceil( ( double ) dimensions[ i ] / downsampleFactor[ i ] ) );

		final long[] offset = new long[ nDim ];
		final int[] actualSize = new int[ nDim ];

		for ( int d = 0; d < nDim; )
		{

			for ( int i = 0; i < nDim; i++ )
				actualSize[ i ] = ( int ) Math.min( parallelSize[ i ] * blockSize[ i ], downsampledDimensions[ i ] - offset[ i ] );

			parallelizeSections.add( new DownsampleBlock( offset.clone(), actualSize.clone() ) );

			for ( d = 0; d < nDim; d++ )
			{
				offset[ d ] += parallelSize[ d ] * blockSize[ d ];
				if ( offset[ d ] < downsampledDimensions[ d ] )
					break;
				else
					offset[ d ] = 0;
			}
		}

		final N5Writer writer = new N5FSWriter( outputGroupName );
		writer.createDataset( outputDatasetName, downsampledDimensions, blockSize, DataType.UINT8, compression );

		final Integer output = sc.parallelize( parallelizeSections )
				.map( new SparkDownsampleFunction( readGroupName, readDatasetName, downsampleFactor, outputGroupName, outputDatasetName ) )
				.reduce( ( i, j ) -> i + j );

		System.out.println( "Across " + parallelizeSections.size() + " parallelized sections, " + output + " cells were downscaled" );
	}
}
