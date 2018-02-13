package bdv.bigcat.util;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.time.DurationFormatUtils;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.google.gson.Gson;

import net.imglib.type.label.FromIntegerTypeConverter;
import net.imglib.type.label.LabelMultisetType;
import net.imglib.type.label.LabelUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class HDFConverter
{

	// TODO make this parallizable/spark and not hdf-to-n5 but convert between
	// various instances of n5 instead
	static public class Parameters
	{
		@Parameter( names = { "--inputfile", "--input", "-i" }, description = "Input HDF5 file" )
		public String inputHDF5 = null;

		@Parameter( names = { "--groupname", "--group", "-g" }, description = "Output group name (N5 group)" )
		public String outputGroupName = null;

		@Parameter( names = { "--datasetname", "--data", "-d" }, description = "Output dataset name (N5 relative path from group)" )
		public String outputDatasetName = null;

		@Parameter( names = { "--cellsize", "-cs" }, description = "Size of cells to use in the output N5 dataset" )
		public List< Integer > cellSize = Arrays.asList( new Integer[] { 64, 64, 8 } );

		@Parameter( names = { "--label", "-l" }, description = "Path to labels in HDF5 file" )
		public String labelPath = "/volumes/labels/neuron_ids";

		@Parameter( names = { "--compression", "-c" }, description = "Compression type to use in output N5 dataset" )
		public String compressionType = "{\"type\":\"raw\"}";

		public boolean init()
		{
			if ( outputGroupName == null || inputHDF5 == null )
				return false;
			if ( outputDatasetName == null )
				outputDatasetName = inputHDF5.contains( ".hdf" ) ? inputHDF5.substring( 0, inputHDF5.lastIndexOf( ".hdf" ) ) : inputHDF5;
			return true;
		}
	}

	public static void main( final String[] args ) throws IOException
	{
		final Parameters params = new Parameters();
		final JCommander jcomm = new JCommander( params, args );

		if ( !params.init() )
		{
			jcomm.usage();
			return;
		}

		final long startTime = System.currentTimeMillis();

		final int[] cellSize = new int[ params.cellSize.size() ];
		Arrays.setAll( cellSize, i -> params.cellSize.get( i ) );

		final N5FSReader reader = new N5FSReader( params.outputGroupName );
		final Gson gson = reader.getGson();
		new N5FSWriter( "/home/phil/local/tmp/123" ).createDataset(
				"456",
				new long[] { 1 },
				new int[] { 1 },
				DataType.UINT8,
				new GzipCompression() );
		final Compression compression = gson.fromJson( params.compressionType, Compression.class );

		convertHDF5toN5(
				params.inputHDF5,
				params.labelPath,
				cellSize,
				params.outputGroupName,
				params.outputDatasetName,
				compression );

		final long endTime = System.currentTimeMillis();

		final String formattedTime = DurationFormatUtils.formatDuration( endTime - startTime, "HH:mm:ss.SSS" );
		System.out.println( "Converted " + params.inputHDF5 + " to N5 dataset at " + params.outputGroupName + " with name " + params.outputDatasetName +
				" in " + formattedTime );
	}

	public static < I extends IntegerType< I > & NativeType< I > > void convertHDF5toN5(
			final String hdf5Path,
			final String labelsPath,
			final int[] cellDimensions,
			final String outputGroupName,
			final String outputDatasetName,
			final Compression compression ) throws IOException
	{
		final N5HDF5Reader hdfReader = new N5HDF5Reader( hdf5Path, cellDimensions );
		final RandomAccessibleInterval< I > img = N5Utils.open( hdfReader, labelsPath );

		final int nDim = img.numDimensions();

		final long[] dimensions = new long[ nDim ];
		img.dimensions( dimensions );

		final N5Writer n5 = new N5FSWriter( outputGroupName );
		n5.createDataset( outputDatasetName, dimensions, cellDimensions, DataType.UINT8, compression );

		final long[] offset = new long[ nDim ];

		final int[] actualCellDimensions = new int[ nDim ];

		for ( int d = 0; d < nDim; )
		{

			Arrays.setAll( actualCellDimensions, i -> ( int ) Math.min( cellDimensions[ i ], dimensions[ i ] - offset[ i ] ) );

			final RandomAccessibleInterval< LabelMultisetType > labelsetImg = Converters.convert(
					img,
					new FromIntegerTypeConverter<>(),
					FromIntegerTypeConverter.geAppropriateType() );
			final IntervalView< LabelMultisetType > tm = Views.offsetInterval( labelsetImg, offset, Arrays.stream( actualCellDimensions ).asLongStream().toArray() );

			final DatasetAttributes attributes = n5.getDatasetAttributes( outputDatasetName );

			final long[] cellOffset = new long[ nDim ];
			for ( int i = 0; i < nDim; i++ )
				cellOffset[ i ] = offset[ i ] / cellDimensions[ i ];

			final ByteArrayDataBlock dataBlock = new ByteArrayDataBlock( actualCellDimensions, cellOffset,
					LabelUtils.serializeLabelMultisetTypes( Views.flatIterable( tm ), ( int ) Intervals.numElements( actualCellDimensions ) ) );

			n5.writeBlock( outputDatasetName, attributes, dataBlock );

			for ( d = 0; d < nDim; d++ )
			{
				offset[ d ] += cellDimensions[ d ];
				if ( offset[ d ] < dimensions[ d ] )
					break;
				else
					offset[ d ] = 0;
			}
		}
	}
}
