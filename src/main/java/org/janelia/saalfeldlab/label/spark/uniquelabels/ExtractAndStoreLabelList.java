package org.janelia.saalfeldlab.label.spark.uniquelabels;

import java.io.IOException;
import java.util.Arrays;

import org.apache.spark.api.java.function.VoidFunction;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import gnu.trove.set.hash.TLongHashSet;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.label.N5CacheLoader;
import net.imglib2.type.label.VolatileLabelMultisetArray;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.util.IntervalIndexer;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import scala.Tuple2;

public class ExtractAndStoreLabelList implements VoidFunction< Tuple2< long[], long[] > >
{

	private final String inputN5;

	private final String outputN5;

	private final String inputDataset;

	private final String outputDataset;

	final long[] dims;

	final int[] blockSize;

	private final boolean isMultisetType;

	public ExtractAndStoreLabelList(
			final String inputN5,
			final String outputN5,
			final String inputDataset,
			final String outputDataset,
			final long[] dims,
			final int[] blockSize,
			final boolean isMultisetType )
	{
		super();
		this.inputN5 = inputN5;
		this.outputN5 = outputN5;
		this.inputDataset = inputDataset;
		this.outputDataset = outputDataset;
		this.dims = dims;
		this.blockSize = blockSize;
		this.isMultisetType = isMultisetType;
	}

	@Override
	public void call( final Tuple2< long[], long[] > minMax ) throws Exception
	{
		final Interval interval = new FinalInterval( minMax._1(), minMax._2() );
		final CellGrid grid = new CellGrid( dims, blockSize );
		final N5Reader n5reader = ExtractUniqueLabelsPerBlock.n5Reader( inputN5, blockSize );
		final TLongHashSet uniqueLabels = new TLongHashSet();
		final RandomAccessibleInterval< ? extends IntegerType< ? > > img = getData(
				interval,
				blockSize,
				grid,
				n5reader,
				inputDataset,
				isMultisetType );
		for ( final IntegerType< ? > val : Views.iterable( img ) )
		{
			uniqueLabels.add( val.getIntegerLong() );
		}
		final long[] pos = Intervals.minAsLongArray( interval );
		Arrays.setAll( pos, d -> pos[ d ] / blockSize[ d ] );
		final LongArrayDataBlock block = new LongArrayDataBlock(
				Intervals.dimensionsAsIntArray( interval ),
				pos,
				uniqueLabels.toArray() );
		final N5Writer n5writer = ExtractUniqueLabelsPerBlock.n5Writer( outputN5, blockSize );
		final DatasetAttributes attributes = new DatasetAttributes(
				grid.getImgDimensions(),
				blockSize,
				DataType.UINT64,
				new GzipCompression() );
		n5writer.writeBlock( outputDataset, attributes, block );
	}

	private static < I extends IntegerType< I > > RandomAccessibleInterval< I > getData(
			final Interval interval,
			final int[] blockSize,
			final CellGrid grid,
			final N5Reader n5reader,
			final String inputDataset,
			final boolean isMultisetType ) throws IOException
	{
		final RandomAccessibleInterval< I > img;
		if ( isMultisetType )
		{
			final N5CacheLoader loader = new N5CacheLoader( n5reader, inputDataset );
			final long[] cellGridPos = ExtractUniqueLabelsPerBlock.blockPos( Intervals.minAsLongArray( interval ), blockSize );
			final Cell< VolatileLabelMultisetArray > array = loader.get( IntervalIndexer.positionToIndex( cellGridPos, grid.getGridDimensions() ) );
			final ArrayImg< LabelMultisetType, VolatileLabelMultisetArray > arrayImg = new ArrayImg<>(
					array.getData(),
					Intervals.dimensionsAsLongArray( interval ),
					new LabelMultisetType().getEntitiesPerPixel() );
			arrayImg.setLinkedType( new LabelMultisetType( arrayImg ) );
			@SuppressWarnings( { "unchecked", "rawtypes" } )
			final RandomAccessibleInterval< I > img2 = ( RandomAccessibleInterval ) arrayImg;
			img = img2;
		}
		else
		{
			img = Views.interval(
					( RandomAccessibleInterval< I > ) N5Utils.open( n5reader, inputDataset ),
					interval );
		}
		return img;
	}

}
