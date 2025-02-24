package org.janelia.saalfeldlab.label.spark.downsample;

import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.algorithm.util.Singleton;
import net.imglib2.cache.CacheLoader;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.ref.SoftRefLoaderCache;
import net.imglib2.cache.util.LoaderCacheAsCacheAdapter;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.img.cell.LazyCellImg.LazyCells;
import net.imglib2.type.label.Label;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.label.LabelMultisetTypeDownscaler;
import net.imglib2.type.label.VolatileLabelMultisetArray;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.message.BasicNameValuePair;
import org.apache.spark.api.java.function.VoidFunction;
import org.janelia.saalfeldlab.label.spark.N5Helpers;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5LabelMultisetCacheLoader;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;

import java.net.URI;
import java.util.Arrays;
import java.util.List;

public class SparkDownsampleFunction implements VoidFunction<Interval> {

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
			final int maxNumEntries) {

		this.inputGroupName = inputGroupName;
		this.inputDatasetName = inputDatasetName;
		this.factor = factor;
		this.outputGroupName = outputGroupName;
		this.outputDatasetName = outputDatasetName;
		this.maxNumEntries = maxNumEntries;
	}

	@Override
	public void call(final Interval interval) throws Exception {

		final URI inputReaderUri = StorageFormat.parseUri(inputGroupName).getB();
		final String inputImgCacheKey = new URIBuilder(inputReaderUri)
				.setParameters(
						new BasicNameValuePair("group", "input"),
						new BasicNameValuePair("type", "reader"),
						new BasicNameValuePair("dataset", inputDatasetName),
						new BasicNameValuePair("call", "spark-downsample-function")
				)
				.toString();

		final CachedCellImg<LabelMultisetType, VolatileLabelMultisetArray> source = Singleton.get(inputImgCacheKey, () -> {
			final N5Reader reader = N5Helpers.n5Reader(inputGroupName);
			final DatasetAttributes attr = reader.getDatasetAttributes(inputDatasetName);

			final long[] sourceDimensions = attr.getDimensions();
			final int[] sourceBlockSize = attr.getBlockSize();

			final N5LabelMultisetCacheLoader cache = new N5LabelMultisetCacheLoader(reader, inputDatasetName, N5LabelMultisetCacheLoader.constantNullReplacement(0));
			return getSource(cache, sourceDimensions, sourceBlockSize);
		});

		final long[] sourceDimensions = source.dimensionsAsLongArray();
		final int nDim = source.numDimensions();

		final long[] blockMinInTarget = Intervals.minAsLongArray(interval);
		final int[] blockSizeInTarget = Intervals.dimensionsAsIntArray(interval);

		final long[] blockMinInSource = new long[nDim];
		final long[] blockMaxInSource = new long[nDim];
		final long[] blockSizeInSource = new long[nDim];
		Arrays.setAll(blockMinInSource, i -> factor[i] * blockMinInTarget[i]);
		Arrays.setAll(blockMaxInSource, i -> Math.min(blockMinInSource[i] + factor[i] * blockSizeInTarget[i] - 1, sourceDimensions[i] - 1));
		Arrays.setAll(blockSizeInSource, i -> blockMaxInSource[i] - blockMinInSource[i] + 1);

		final CellGrid sourceGrid = source.getCellGrid();
		final int[] sourceCellDimensions = new int[sourceGrid.numDimensions()];
		Arrays.setAll(sourceCellDimensions, sourceGrid::cellDimension);
		final List<long[]> cellPositions = Grids.collectAllOffsets(
				blockMinInSource,
				blockMaxInSource,
				sourceCellDimensions);

		final LazyCells<Cell<VolatileLabelMultisetArray>> cells = source.getCells();
		final RandomAccess<Cell<VolatileLabelMultisetArray>> cellsAccess = cells.randomAccess();
		for (final long[] pos : cellPositions) {
			Arrays.setAll(pos, d -> pos[d] / sourceCellDimensions[d]);
			cellsAccess.setPosition(pos);
		}

		final URI outputWriterUri = StorageFormat.parseUri(outputGroupName).getB();
		final String outputWriterCacheKey = new URIBuilder(outputWriterUri)
				.setParameters(
						new BasicNameValuePair("group", "output"),
						new BasicNameValuePair("type", "writer"),
						new BasicNameValuePair("dataset", outputDatasetName),
						new BasicNameValuePair("call", "spark-downsample-function")
				)
				.toString();

		final N5Writer writer = Singleton.get(outputWriterCacheKey, () -> N5Helpers.n5Writer(outputGroupName));
		final DatasetAttributes writerAttributes = writer.getDatasetAttributes(outputDatasetName);

		final long[] writeLocation = new long[nDim];

		// Hopefully, the block size of this is consistent with the size of blockSizeInTarget
		final VolatileLabelMultisetArray downscaledCell = LabelMultisetTypeDownscaler.createDownscaledCell(
				Views.offsetInterval(source, blockMinInSource, blockSizeInSource),
				factor,
				maxNumEntries);

		final byte[] bytes = new byte[LabelMultisetTypeDownscaler.getSerializedVolatileLabelMultisetArraySize(downscaledCell)];
		LabelMultisetTypeDownscaler.serializeVolatileLabelMultisetArray(downscaledCell, bytes);

		for (int i = 0; i < nDim; i++) {
			writeLocation[i] = blockMinInTarget[i] / writerAttributes.getBlockSize()[i];
		}

		final ByteArrayDataBlock dataBlock = new ByteArrayDataBlock(blockSizeInTarget, writeLocation, bytes);

		// Empty blocks will not be written out.
		// Delete blocks to avoid remnant blocks if overwriting.
		writer.deleteBlock(outputDatasetName, writeLocation);

		/* Since imglib2-labelmultisets-0.15.0 if the backing lists are all `0` then the storage array is empty
		* 	so we only need to write the block if there is data to do so. No need for an explicit foreground check.  */
		if (downscaledCell.getCurrentStorageArray().length != 0)
			writer.writeBlock(outputDatasetName, writerAttributes, dataBlock);
	}

	public static CachedCellImg<LabelMultisetType, VolatileLabelMultisetArray> getSource(
			final CacheLoader<Long, Cell<VolatileLabelMultisetArray>> cacheLoader,
			final long[] dimensions,
			final int[] blockSize) {

		final SoftRefLoaderCache<Long, Cell<VolatileLabelMultisetArray>> cache = new SoftRefLoaderCache<>();
		final LoaderCacheAsCacheAdapter<Long, Cell<VolatileLabelMultisetArray>> wrappedCache = new LoaderCacheAsCacheAdapter<>(cache, cacheLoader);

		final CachedCellImg<LabelMultisetType, VolatileLabelMultisetArray> source = new CachedCellImg<>(
				new CellGrid(dimensions, blockSize),
				new LabelMultisetType().getEntitiesPerPixel(),
				wrappedCache,
				new VolatileLabelMultisetArray(0, true, new long[]{Label.INVALID}));
		source.setLinkedType(new LabelMultisetType(source));
		return source;
	}
}
