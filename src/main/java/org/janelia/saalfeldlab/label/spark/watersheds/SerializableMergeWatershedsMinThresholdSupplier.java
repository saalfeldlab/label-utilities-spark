package org.janelia.saalfeldlab.label.spark.watersheds;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.unionfind.IntArrayUnionFind;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

import java.io.Serializable;
import java.util.function.LongUnaryOperator;
import java.util.function.Supplier;

class SerializableMergeWatershedsMinThresholdSupplier implements Supplier<MergeWatersheds>, Serializable {

	private final double threshold;

	SerializableMergeWatershedsMinThresholdSupplier(final double threshold) {
		this.threshold = threshold;
	}

	private final class MW implements MergeWatersheds, Serializable {

		@Override
		public <T extends RealType<T>> LongUnaryOperator getMapping(
				final RandomAccessibleInterval<T> relief,
				final RandomAccessibleInterval<UnsignedLongType> labels,
				final long maxId) {
			final IntArrayUnionFind uf = new IntArrayUnionFind((int) (maxId + 1));
			for (int d = 0; d < relief.numDimensions(); ++d) {
				final long[] min1 = Intervals.minAsLongArray(relief);
				final long[] max1 = Intervals.maxAsLongArray(relief);
				final long[] min2 = min1.clone();
				final long[] max2 = max1.clone();
				max1[d] -= 1;
				min2[d] += 1;
				final Cursor<T> reliefCursor1 = Views.flatIterable(Views.interval(relief, min1, max1)).cursor();
				final Cursor<T> reliefCursor2 = Views.flatIterable(Views.interval(relief, min2, max2)).cursor();
				final Cursor<UnsignedLongType> labelsCursor1 = Views.flatIterable(Views.interval(labels, min1, max1)).cursor();
				final Cursor<UnsignedLongType> labelsCursor2 = Views.flatIterable(Views.interval(labels, min2, max2)).cursor();
				while (reliefCursor1.hasNext()) {
					reliefCursor1.fwd();
					reliefCursor2.fwd();
					labelsCursor1.fwd();
					labelsCursor2.fwd();
					if (reliefCursor1.get().getRealDouble() > threshold && reliefCursor2.get().getRealDouble() > threshold){
						final long r1 = uf.findRoot(labelsCursor1.get().getIntegerLong());
						final long r2 = uf.findRoot(labelsCursor2.get().getIntegerLong());
						if (r1 != r2 && r1 != 0 && r2 != 0)
							uf.join(r1, r2);
					}
				}
			}

			return uf::findRoot;
		}
	}

	@Override
	public MergeWatersheds get() {
		return new MW();
	}
}
