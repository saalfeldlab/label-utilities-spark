package org.janelia.saalfeldlab.label.spark.uniquelabels;

import gnu.trove.set.hash.TLongHashSet;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.Singleton;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.NativeType;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.label.VolatileLabelMultisetArray;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.util.IntervalIndexer;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.message.BasicNameValuePair;
import org.apache.spark.api.java.function.Function;
import org.janelia.saalfeldlab.label.spark.N5Helpers;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5LabelMultisetCacheLoader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;
import scala.Tuple2;

import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.function.Supplier;

public class ExtractAndStoreLabelList implements Function<Tuple2<long[], long[]>, Long> {

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
			final boolean isMultisetType) {

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
	public Long call(final Tuple2<long[], long[]> minMax) throws Exception {

		return callImpl(dims, blockSize, inputN5, inputDataset, outputN5, outputDataset, isMultisetType, minMax);
	}

	private static <I extends IntegerType<I> & NativeType<I>> RandomAccessibleInterval<I> getData(
			final Interval interval,
			final int[] blockSize,
			final CellGrid grid,
			final String n5ReaderLocation,
			final String inputDataset,
			final boolean isMultisetType) throws Exception {

		final URI containerUri = StorageFormat.parseUri(n5ReaderLocation).getB();

		final String imgCacheKey = new URIBuilder(containerUri)
				.setParameters(
						new BasicNameValuePair("call", "extract-and-store-label-list-get-data"),
						new BasicNameValuePair("dataset", inputDataset),
						new BasicNameValuePair("blockSize", Arrays.toString(blockSize)),
						new BasicNameValuePair("isMultiset", "" + isMultisetType)
				).toString();

			if (isMultisetType) {
				final N5LabelMultisetCacheLoader loader = Singleton.get(imgCacheKey, () -> {
					final N5Reader n5Reader = ExtractUniqueLabelsPerBlock.n5Reader(n5ReaderLocation, blockSize);
					return new N5LabelMultisetCacheLoader(n5Reader, inputDataset, N5LabelMultisetCacheLoader.constantNullReplacement(0));
				});
				final long[] cellGridPos = ExtractUniqueLabelsPerBlock.blockPos(Intervals.minAsLongArray(interval), blockSize);
				final Cell<VolatileLabelMultisetArray> array = loader.get(IntervalIndexer.positionToIndex(cellGridPos, grid.getGridDimensions()));
				final ArrayImg<LabelMultisetType, VolatileLabelMultisetArray> arrayImg = new ArrayImg<>(
						array.getData(),
						Intervals.dimensionsAsLongArray(interval),
						new LabelMultisetType().getEntitiesPerPixel());
				arrayImg.setLinkedType(new LabelMultisetType(arrayImg));
				@SuppressWarnings({"unchecked", "rawtypes"}) final RandomAccessibleInterval<I> img2 = (RandomAccessibleInterval)arrayImg;
				return img2;
			} else {
				final CachedCellImg<I, ?> img = Singleton.get(imgCacheKey, () -> {
					final N5Reader n5Reader = ExtractUniqueLabelsPerBlock.n5Reader(n5ReaderLocation, blockSize);
					return N5Helpers.openBounded(n5Reader, inputDataset);
				});
				return Views.interval(img, interval);
			}
	}

	private static <I extends IntegerType<I> & NativeType<I>> long callImpl(
			final long[] dims,
			final int[] blockSize,
			final String inputN5,
			final String inputDataset,
			final String outputN5,
			final String outputDataset,
			final boolean isMultisetType,
			final Tuple2<long[], long[]> minMax) throws Exception {

		final Interval interval = new FinalInterval(minMax._1(), minMax._2());
		final CellGrid grid = new CellGrid(dims, blockSize);
		final TLongHashSet uniqueLabels = new TLongHashSet();
		final RandomAccessibleInterval<I> img = getData(
				interval,
				blockSize,
				grid,
				inputN5,
				inputDataset,
				isMultisetType);
		long maxVal = 0;
		for (final I val : Views.iterable(img)) {
			final long primitiveVal = val.getIntegerLong();
			uniqueLabels.add(primitiveVal);
			if (primitiveVal > maxVal)
				maxVal = primitiveVal;
		}
		final long[] pos = Intervals.minAsLongArray(interval);
		Arrays.setAll(pos, d -> pos[d] / blockSize[d]);
		final LongArrayDataBlock block = new LongArrayDataBlock(
				Intervals.dimensionsAsIntArray(interval),
				pos,
				uniqueLabels.toArray());

		final URI outputUri = StorageFormat.parseUri(outputN5).getB();
		final String writerCacheKey = new URIBuilder(outputUri)
				.setParameters(
						new BasicNameValuePair("call", "extract-and-store-label-list-call-impl"),
						new BasicNameValuePair("type", "writer"),
						new BasicNameValuePair("blockSize", Arrays.toString(blockSize)),
						new BasicNameValuePair("isMultiset", "" + isMultisetType)
				).toString();

		final N5Writer n5writer = Singleton.get(writerCacheKey, () -> ExtractUniqueLabelsPerBlock.n5Writer(outputN5, blockSize));
		final DatasetAttributes attributes = new DatasetAttributes(
				grid.getImgDimensions(),
				blockSize,
				DataType.UINT64,
				new GzipCompression());
		n5writer.writeBlock(outputDataset, attributes, block);
		return maxVal;
	}

}
