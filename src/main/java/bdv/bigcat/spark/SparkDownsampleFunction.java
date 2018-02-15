package bdv.bigcat.spark;

import java.util.Arrays;

import org.apache.spark.api.java.function.VoidFunction;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.CacheLoader;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.ref.SoftRefLoaderCache;
import net.imglib2.cache.util.LoaderCacheAsCacheAdapter;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.label.Label;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.label.LabelMultisetTypeDownscaler;
import net.imglib2.type.label.LabelUtils;
import net.imglib2.type.label.Multiset.Entry;
import net.imglib2.type.label.N5CacheLoader;
import net.imglib2.type.label.VolatileLabelMultisetArray;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

public class SparkDownsampleFunction implements VoidFunction< Interval >
{

	private static final long serialVersionUID = 1384028449836651390L;

	private final String inputGroupName;

	private final String inputDatasetName;

	private final int[] factor;

	private final String outputGroupName;

	private final String outputDatasetName;

	public SparkDownsampleFunction(
			final String inputGroupName,
			final String inputDatasetName,
			final int[] factor,
			final String outputGroupName,
			final String outputDatasetName )
	{
		this.inputGroupName = inputGroupName;
		this.inputDatasetName = inputDatasetName;
		this.factor = factor;
		this.outputGroupName = outputGroupName;
		this.outputDatasetName = outputDatasetName;
	}

	@Override
	public void call( final Interval interval ) throws Exception
	{

		final N5Reader reader = new N5FSReader( inputGroupName );
		final DatasetAttributes attr = reader.getDatasetAttributes( inputDatasetName );

		final long[] sourceDimensions = attr.getDimensions();
		final int[] sourceBlockSize = attr.getBlockSize();
		final int nDim = attr.getNumDimensions();

		final long[] blockMinInTarget = Intervals.minAsLongArray( interval );
		final int[] blockSizeInTarget = Intervals.dimensionsAsIntArray( interval );

		final long[] blockMinInSource = new long[ nDim ];
		final long[] blockMaxInSource = new long[ nDim ];
		final long[] blockSizeInSource = new long[ nDim ];
		Arrays.setAll( blockMinInSource, i -> factor[ i ] * blockMinInTarget[ i ] );
		Arrays.setAll( blockMaxInSource, i -> Math.min( blockMinInSource[ i ] + factor[ i ] * blockSizeInTarget[ i ] - 1, sourceDimensions[ i ] - 1 ) );
		Arrays.setAll( blockSizeInSource, i -> blockMaxInSource[ i ] - blockMinInSource[ i ] + 1 );

		final RandomAccessibleInterval< LabelMultisetType > source = getSource( new N5CacheLoader( reader, inputDatasetName ), sourceDimensions, sourceBlockSize );

		// TODO Should this be passed from outside? Needs to load one additional
		// block for (almost) all tasks
		int eachCount = 0;
		for ( final Entry< Label > e : Util.getTypeFromInterval( source ).entrySet() )
			eachCount += e.getCount();
		final RandomAccessible< LabelMultisetType > extendedImg = Views.extendValue( source, LabelUtils.getOutOfBounds( eachCount ) );

		// Hopefully, the block size of this is consistent with the size of
		// blockSizeInTarget
		final N5Writer writer = new N5FSWriter( outputGroupName );
		final DatasetAttributes writerAttributes = writer.getDatasetAttributes( outputDatasetName );

		final long[] writeLocation = new long[ nDim ];

		final VolatileLabelMultisetArray downscaledCell = LabelMultisetTypeDownscaler.createDownscaledCell( Views.offsetInterval( extendedImg, blockMinInSource, blockSizeInSource ), factor );

		final byte[] bytes = new byte[ LabelMultisetTypeDownscaler.getSerializedVolatileLabelMultisetArraySize( downscaledCell ) ];
		LabelMultisetTypeDownscaler.serializeVolatileLabelMultisetArray( downscaledCell, bytes );

		for ( int i = 0; i < nDim; i++ )
			writeLocation[ i ] = blockMinInTarget[ i ] / writerAttributes.getBlockSize()[ i ];

		final ByteArrayDataBlock dataBlock = new ByteArrayDataBlock( blockSizeInTarget, writeLocation, bytes );
		writer.writeBlock( outputDatasetName, writerAttributes, dataBlock );

	}

	public static RandomAccessibleInterval< LabelMultisetType > getSource(
			final CacheLoader< Long, Cell< VolatileLabelMultisetArray > > cacheLoader,
			final long[] dimensions,
			final int[] blockSize )
	{
		final SoftRefLoaderCache< Long, Cell< VolatileLabelMultisetArray > > cache = new SoftRefLoaderCache<>();
		final LoaderCacheAsCacheAdapter< Long, Cell< VolatileLabelMultisetArray > > wrappedCache = new LoaderCacheAsCacheAdapter<>( cache, cacheLoader );

		final CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray > source = new CachedCellImg<>(
				new CellGrid( dimensions, blockSize ), new LabelMultisetType(), wrappedCache, new VolatileLabelMultisetArray( 0, true ) );
		return source;
	}
}
