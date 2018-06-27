package org.janelia.saalfeldlab.multisets.spark.downsample;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.function.VoidFunction;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.cache.CacheLoader;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.ref.SoftRefLoaderCache;
import net.imglib2.cache.util.LoaderCacheAsCacheAdapter;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.img.cell.LazyCellImg.LazyCells;
import net.imglib2.type.label.Label;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.label.LabelMultisetType.Entry;
import net.imglib2.type.label.LabelMultisetTypeDownscaler;
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

	private final int maxNumEntries;

	public SparkDownsampleFunction(
			final String inputGroupName,
			final String inputDatasetName,
			final int[] factor,
			final String outputGroupName,
			final String outputDatasetName,
			final int maxNumEntries )
	{
		this.inputGroupName = inputGroupName;
		this.inputDatasetName = inputDatasetName;
		this.factor = factor;
		this.outputGroupName = outputGroupName;
		this.outputDatasetName = outputDatasetName;
		this.maxNumEntries = maxNumEntries;
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

		final CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray > source =
				getSource( new N5CacheLoader( reader, inputDatasetName ), sourceDimensions, sourceBlockSize );
		final CellGrid sourceGrid = source.getCellGrid();
		final int[] sourceCellDimensions = new int[ sourceGrid.numDimensions() ];
		Arrays.setAll( sourceCellDimensions, sourceGrid::cellDimension );
		final List< long[] > cellPositions = Grids.collectAllOffsets(
				blockMinInSource,
				blockMaxInSource,
				sourceCellDimensions );

		final LazyCells< Cell< VolatileLabelMultisetArray > > cells = source.getCells();
		final RandomAccess< Cell< VolatileLabelMultisetArray > > cellsAccess = cells.randomAccess();
		for ( final long[] pos : cellPositions )
		{
			Arrays.setAll( pos, d -> pos[ d ] / sourceCellDimensions[ d ] );
			cellsAccess.setPosition( pos );
		}
		source.getCellGrid();

		// TODO Should this be passed from outside? Needs to load one additional
		// block for (almost) all tasks
		int eachCount = 0;
		for ( final Entry< Label > e : Util.getTypeFromInterval( source ).entrySet() )
		{
			eachCount += e.getCount();
		}

		// Hopefully, the block size of this is consistent with the size of
		// blockSizeInTarget
		final N5Writer writer = new N5FSWriter( outputGroupName );
		final DatasetAttributes writerAttributes = writer.getDatasetAttributes( outputDatasetName );

		final long[] writeLocation = new long[ nDim ];

		final VolatileLabelMultisetArray downscaledCell = LabelMultisetTypeDownscaler
				.createDownscaledCell(
						Views.offsetInterval( source, blockMinInSource, blockSizeInSource ),
						factor,
						maxNumEntries );

		final byte[] bytes = new byte[ LabelMultisetTypeDownscaler.getSerializedVolatileLabelMultisetArraySize( downscaledCell ) ];
		LabelMultisetTypeDownscaler.serializeVolatileLabelMultisetArray( downscaledCell, bytes );

		for ( int i = 0; i < nDim; i++ )
		{
			writeLocation[ i ] = blockMinInTarget[ i ] / writerAttributes.getBlockSize()[ i ];
		}

		final ByteArrayDataBlock dataBlock = new ByteArrayDataBlock( blockSizeInTarget, writeLocation, bytes );
		writer.writeBlock( outputDatasetName, writerAttributes, dataBlock );

	}

	public static CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray > getSource(
			final CacheLoader< Long, Cell< VolatileLabelMultisetArray > > cacheLoader,
			final long[] dimensions,
			final int[] blockSize )
	{
		final SoftRefLoaderCache< Long, Cell< VolatileLabelMultisetArray > > cache = new SoftRefLoaderCache<>();
		final LoaderCacheAsCacheAdapter< Long, Cell< VolatileLabelMultisetArray > > wrappedCache = new LoaderCacheAsCacheAdapter<>( cache, cacheLoader );

		final CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray > source = new CachedCellImg<>(
				new CellGrid( dimensions, blockSize ),
				new LabelMultisetType().getEntitiesPerPixel(),
				wrappedCache,
				new VolatileLabelMultisetArray( 0, true, new long[] { Label.INVALID } ) );
		source.setLinkedType( new LabelMultisetType( source ) );
		return source;
	}
}
