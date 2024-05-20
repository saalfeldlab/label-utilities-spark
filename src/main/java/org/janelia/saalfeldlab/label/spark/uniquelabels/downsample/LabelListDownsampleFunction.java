package org.janelia.saalfeldlab.label.spark.uniquelabels.downsample;

import gnu.trove.set.hash.TLongHashSet;
import net.imglib2.Interval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.util.Intervals;
import org.apache.spark.api.java.function.VoidFunction;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import java.util.Arrays;
import java.util.List;

public class LabelListDownsampleFunction implements VoidFunction<Interval> {

	private static final long serialVersionUID = 1384028449836651390L;

	private final String inputGroupName;

	private final String inputDatasetName;

	private final int[] factor;

	private final String outputGroupName;

	private final String outputDatasetName;

	public LabelListDownsampleFunction(
			final String inputGroupName,
			final String inputDatasetName,
			final int[] factor,
			final String outputGroupName,
			final String outputDatasetName) {

		this.inputGroupName = inputGroupName;
		this.inputDatasetName = inputDatasetName;
		this.factor = factor;
		this.outputGroupName = outputGroupName;
		this.outputDatasetName = outputDatasetName;
	}

	@Override
	public void call(final Interval interval) throws Exception {

		final N5Reader reader = new N5FSReader(inputGroupName);
		final DatasetAttributes attr = reader.getDatasetAttributes(inputDatasetName);

		final long[] sourceDimensions = attr.getDimensions();
		final int nDim = attr.getNumDimensions();

		final long[] blockMinInTarget = Intervals.minAsLongArray(interval);
		final int[] blockSizeInTarget = Intervals.dimensionsAsIntArray(interval);

		final long[] blockMinInSource = new long[nDim];
		final long[] blockMaxInSource = new long[nDim];
		final long[] blockSizeInSource = new long[nDim];
		Arrays.setAll(blockMinInSource, i -> factor[i] * blockMinInTarget[i]);
		Arrays.setAll(blockMaxInSource, i -> Math.min(blockMinInSource[i] + factor[i] * blockSizeInTarget[i] - 1, sourceDimensions[i] - 1));
		Arrays.setAll(blockSizeInSource, i -> blockMaxInSource[i] - blockMinInSource[i] + 1);

		final List<long[]> cellPositions = Grids.collectAllOffsets(
				blockMinInSource,
				blockMaxInSource,
				attr.getBlockSize());

		final TLongHashSet containedLabels = new TLongHashSet();
		final int[] bs = attr.getBlockSize();
		for (final long[] cellPos : cellPositions) {
			Arrays.setAll(cellPos, d -> cellPos[d] / bs[d]);
			final LongArrayDataBlock source = (LongArrayDataBlock)reader.readBlock(inputDatasetName, attr, cellPos);
			containedLabels.addAll(source.getData());
		}
		;

		final N5Writer writer = new N5FSWriter(outputGroupName);
		final DatasetAttributes writerAttributes = writer.getDatasetAttributes(outputDatasetName);

		final long[] writeLocation = new long[nDim];

		for (int i = 0; i < nDim; i++) {
			writeLocation[i] = blockMinInTarget[i] / writerAttributes.getBlockSize()[i];
		}

		final LongArrayDataBlock dataBlock = new LongArrayDataBlock(blockSizeInTarget, writeLocation, containedLabels.toArray());
		writer.writeBlock(outputDatasetName, writerAttributes, dataBlock);

	}
}
