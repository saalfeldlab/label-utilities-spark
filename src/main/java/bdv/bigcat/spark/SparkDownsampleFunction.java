package bdv.bigcat.spark;

import java.util.Arrays;

import org.apache.spark.api.java.function.Function;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import net.imglib.type.label.Label;
import net.imglib.type.label.LabelMultisetType;
import net.imglib.type.label.LabelMultisetTypeDownscaler;
import net.imglib.type.label.LabelUtils;
import net.imglib.type.label.Multiset.Entry;
import net.imglib.type.label.N5CacheLoader;
import net.imglib.type.label.VolatileLabelMultisetArray;
import net.imglib2.RandomAccessible;
import net.imglib2.cache.CacheLoader;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.ref.BoundedSoftRefLoaderCache;
import net.imglib2.cache.util.LoaderCacheAsCacheAdapter;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.view.Views;

public class SparkDownsampleFunction implements Function< DownsampleBlock, Integer >
{

	private static final long serialVersionUID = 1384028449836651390L;

	private final String inputGroupName;

	private final String inputDatasetName;

	private final int[] factor;

	private final String outputGroupName;

	private final String outputDatasetName;

	public SparkDownsampleFunction( final String inputGroupName, final String inputDatasetName, final int[] factor, final String outputGroupName, final String outputDatasetName )
	{
		this.inputGroupName = inputGroupName;
		this.inputDatasetName = inputDatasetName;
		this.factor = factor;
		this.outputGroupName = outputGroupName;
		this.outputDatasetName = outputDatasetName;
	}

	@Override
	public Integer call( final DownsampleBlock targetRegion ) throws Exception
	{

		final N5Reader reader = new N5FSReader( inputGroupName );
		final DatasetAttributes attr = reader.getDatasetAttributes( inputDatasetName );

		final long[] dimensions = attr.getDimensions();
		final int[] blocksize = attr.getBlockSize();

		final int nDim = dimensions.length;
		final long[] offset = new long[ nDim ];

		final int[] targetSize = targetRegion.getSize();
		final long[] targetMin = targetRegion.getMin();

		final long[] actualLocation = new long[ nDim ];
		final long[] actualSize = new long[ nDim ];

		final CacheLoader< Long, Cell< VolatileLabelMultisetArray > > cacheLoader = new N5CacheLoader( reader, inputDatasetName );

		final BoundedSoftRefLoaderCache< Long, Cell< VolatileLabelMultisetArray > > cache = new BoundedSoftRefLoaderCache<>( 1 );
		final LoaderCacheAsCacheAdapter< Long, Cell< VolatileLabelMultisetArray > > wrappedCache = new LoaderCacheAsCacheAdapter<>( cache, cacheLoader );

		final CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray > inputImg = new CachedCellImg<>(
				new CellGrid( dimensions, blocksize ), new LabelMultisetType(), wrappedCache, new VolatileLabelMultisetArray( 0, true ) );

		int eachCount = 0;
		for ( final Entry< Label > e : inputImg.firstElement().entrySet() )
			eachCount += e.getCount();

		final RandomAccessible< LabelMultisetType > extendedImg = Views.extendValue( inputImg, LabelUtils.getOutOfBounds( eachCount ) );

		VolatileLabelMultisetArray downscaledCell;

		int numCellsDownscaled = 0;

		final N5Writer writer = new N5FSWriter( outputGroupName );
		final DatasetAttributes writerAttributes = writer.getDatasetAttributes( outputDatasetName );

		final long[] writeLocation = new long[ nDim ];

		for ( int d = 0; d < nDim; )
		{

			Arrays.setAll( actualLocation, i -> factor[ i ] * ( targetMin[ i ] + offset[ i ] ) );

			// TODO: figure out what part of this is redundant, if any, and
			// clarify it
			Arrays.setAll( actualSize, i -> Math.min(
					factor[ i ] * ( offset[ i ] + blocksize[ i ] > targetMin[ i ] + targetSize[ i ] ? targetMin[ i ] + targetSize[ i ] - offset[ i ] : blocksize[ i ] ),
					factor[ i ] * ( int ) Math.ceil( ( dimensions[ i ] - actualLocation[ i ] ) / ( double ) factor[ i ] ) ) );

			downscaledCell = LabelMultisetTypeDownscaler.createDownscaledCell( Views.offsetInterval( extendedImg, actualLocation, actualSize ), factor );

			final byte[] bytes = new byte[ LabelMultisetTypeDownscaler.getSerializedVolatileLabelMultisetArraySize( downscaledCell ) ];
			LabelMultisetTypeDownscaler.serializeVolatileLabelMultisetArray( downscaledCell, bytes );

			for ( int i = 0; i < nDim; i++ )
				writeLocation[ i ] = ( targetMin[ i ] + offset[ i ] ) / blocksize[ i ];

			final ByteArrayDataBlock dataBlock = new ByteArrayDataBlock( blocksize, writeLocation, bytes );
			writer.writeBlock( outputDatasetName, writerAttributes, dataBlock );

			numCellsDownscaled++;

			for ( d = 0; d < nDim; d++ )
			{
				offset[ d ] += blocksize[ d ];
				if ( offset[ d ] < targetSize[ d ] )
					break;
				else
					offset[ d ] = 0;
			}
		}

		return numCellsDownscaled;
	}
}
