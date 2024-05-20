package org.janelia.saalfeldlab.label.spark.watersheds;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TLongObjectMap;
import gnu.trove.map.hash.TLongObjectHashMap;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.unionfind.IntArrayUnionFind;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

import java.io.Serializable;
import java.util.function.LongUnaryOperator;
import java.util.function.Supplier;

class SerializableMergeWatershedsMedianThresholdSupplier implements Supplier<MergeWatersheds>, Serializable {

	private final double threshold;

	public SerializableMergeWatershedsMedianThresholdSupplier(final double threshold) {

		this.threshold = threshold;
	}

	private final class MW implements MergeWatersheds, Serializable {

		@Override
		public <T extends RealType<T>> LongUnaryOperator getMapping(
				final RandomAccessibleInterval<T> relief,
				final RandomAccessibleInterval<UnsignedLongType> labels,
				final long maxId) {

			final TLongObjectMap<TLongObjectMap<TDoubleArrayList>> edgeAffinities = new TLongObjectHashMap<>();

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

					long l1 = labelsCursor1.get().getIntegerLong();
					long l2 = labelsCursor2.get().getIntegerLong();

					if (l1 == l2 || l1 == 0 || l2 == 0)
						continue;

					if (l1 > l2) {
						final long tmp = l2;
						l2 = l1;
						l1 = tmp;
					}

					final double a1 = reliefCursor1.get().getRealDouble();
					final double a2 = reliefCursor1.get().getRealDouble();

					if (!edgeAffinities.containsKey(l1))
						edgeAffinities.put(l1, new TLongObjectHashMap<>());

					final TLongObjectMap<TDoubleArrayList> map = edgeAffinities.get(l1);

					if (!map.containsKey(l2))
						map.put(l2, new TDoubleArrayList());

					final TDoubleArrayList al = map.get(l2);

					al.add(a1);
					al.add(a2);
				}
			}

			final IntArrayUnionFind uf = new IntArrayUnionFind((int)(maxId + 1));

			edgeAffinities.forEachEntry((l1, map) -> {
				map.forEachEntry((l2, affinities) -> {
					affinities.sort();
					final double median = affinities.get(affinities.size() / 2);
					if (median > threshold) {
						final long r1 = uf.findRoot(l1);
						final long r2 = uf.findRoot(l2);
						uf.join(r1, r2);
					}
					return true;
				});
				return true;
			});
			return uf::findRoot;
		}
	}

	@Override
	public MergeWatersheds get() {

		return new MW();
	}
}
