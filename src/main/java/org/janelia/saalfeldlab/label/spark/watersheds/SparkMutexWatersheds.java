package org.janelia.saalfeldlab.label.spark.watersheds;

import gnu.trove.iterator.TLongLongIterator;
import gnu.trove.list.TLongList;
import gnu.trove.list.array.TLongArrayList;
import gnu.trove.map.TLongLongMap;
import gnu.trove.map.TLongObjectMap;
import gnu.trove.map.hash.TLongLongHashMap;
import gnu.trove.map.hash.TLongObjectHashMap;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.algorithm.util.unionfind.LongHashMapUnionFind;
import net.imglib2.algorithm.util.unionfind.UnionFind;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.ConstantUtils;
import net.imglib2.util.IntervalIndexer;
import net.imglib2.util.Intervals;
import net.imglib2.util.StopWatch;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.imglib2.mutex.MutexWatershed;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import picocli.CommandLine;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SparkMutexWatersheds {

	@CommandLine.Command(name = "spark-mutex-watersheds-cremi")
	public static class CREMI implements Callable<Integer> {

		@CommandLine.Option(names = {"--samples", "-s"}, split = ",", defaultValue = "A,B,C,0,1,2", showDefaultValue = CommandLine.Help.Visibility.ALWAYS)
		String[] samples = null;

		@CommandLine.Option(names = {"--setups", "-S"}, split = ",", defaultValue = "0,3", showDefaultValue = CommandLine.Help.Visibility.ALWAYS)
		int[] setups = null;

		@CommandLine.Option(names = {"--iterations", "-i"}, split = ",", defaultValue = "500000", showDefaultValue = CommandLine.Help.Visibility.ALWAYS)
		int[] iterations = null;

		@CommandLine.Option(names = {"--block-size", "-b"}, split = ",", defaultValue = "64,64,64", showDefaultValue = CommandLine.Help.Visibility.ALWAYS)
		int[] blockSize = null;

		@CommandLine.Option(names = {"--blocks-per-task", "-p"}, split = ",", defaultValue = "1,1,1", showDefaultValue = CommandLine.Help.Visibility.ALWAYS)
		int[] blocksPerTask = null;

		@CommandLine.Option(names = {"--super-blocks-per-task", "-P"}, split = ",", defaultValue = "3,3,3", showDefaultValue = CommandLine.Help.Visibility.ALWAYS)
		int[] superBlocksPerTask = null;

		@CommandLine.Option(names = {"--threshold", "-t"})
		Double threshold = null;

		@CommandLine.Option(names = {"--overwrite-existing"})
		boolean overwriteExisting = false;



		public static void main(String... args) {
			new CommandLine(new CREMI()).execute(args);
		}

		@Override
		public Integer call() throws IOException {

			if (blockSize.length == 1)
				blockSize = new int[] {blockSize[0], blockSize[0], blockSize[0]};

			if (blocksPerTask.length == 1)
				blocksPerTask = new int[] {blocksPerTask[0], blocksPerTask[0], blocksPerTask[0]};

			if (superBlocksPerTask.length == 1)
				superBlocksPerTask = new int[] {superBlocksPerTask[0], superBlocksPerTask[0], superBlocksPerTask[0]};


			final SparkConf conf = new SparkConf().setAppName(CREMI.class.getName());
			try (final JavaSparkContext sc = new JavaSparkContext(conf)) {
				for (final int iteration : iterations) {
					for (final int setup : setups) {
						final String basePath = basePathFor(setup, iteration);
						for (final String sample: samples) {
							final String containerPath = containerPathFor(sample);
							final String gliaDataset = gliaFrom(basePath);
							final long[] outputSize = Intervals.dimensionsAsLongArray(N5Utils.open(new N5FSReader(containerPath), gliaDataset));
							final StopWatch sw = StopWatch.createAndStart();
							runMutexWatersheds(
									sc,
									containerPath,
									affinitiesFrom(basePath),
									MASK_DATASET,
									gliaDataset,
									mutexWatershedDatasetFrom(basePath, threshold),
									mutexWatershedRelabeledDatasetFrom(basePath, threshold),
									mutexWatershedMergedDatasetFrom(basePath, threshold),
									threshold,
									outputSize,
									blockSize,
									blocksPerTask,
									superBlocksPerTask,
									overwriteExisting,
									OFFSETS);
							sw.stop();
							System.out.println(String.format(
									"Ran mutex watersheds for CREMI sample=%s setup=%d iteration=%d in %s.",
									sample,
									setup,
									iteration,
									StopWatch.secondsToString(sw.seconds())));
						}
					}
				}
			}
			return 0;
		}

		private static final String CONTAINER_PATTERN = "/nrs/saalfeld/hanslovskyp/experiments/quasi-isotropic-predictions/affinities-glia/neuron_ids-unlabeled-unmask-background/predictions/CREMI/sample_%s.n5";

		private static final String BASE_PATH_PATTERN = "volumes/predictions/neuron_ids-unlabeled-unmask-background/%d/%d";

		private static final String MASK_DATASET = "volumes/masks/prediction-mask";

		private static final long[][] OFFSETS = {
				{-1, 0, 0}, {-2, 0, 0}, {-5, 0, 0}, {-10, 0, 0},
				{0, -1, 0}, {0, -2, 0}, {0, -5, 0}, {0, -10, 0},
				{0, 0, -1}, {0, 0, -2}, {0, 0, -5}, {0, 0, -10}};

		private static String containerPathFor(final String sample) {
			return String.format(CONTAINER_PATTERN, sample);
		}

		private static String basePathFor(final int setup, final int iteration) {
			return String.format(BASE_PATH_PATTERN, setup, iteration);
		}

		private static String datasetFrom(final String basePath, final String datasetName) {
			return String.format("%s/%s", basePath, datasetName);
		}

		private static String affinitiesFrom(final String basePath) {
			return datasetFrom(basePath, "affinities");
		}

		private static String gliaFrom(final String basePath) {
			return datasetFrom(basePath, "glia");
		}

		private static String mutexWatershedDatasetFrom(final String basePath, final Double threshold) {
			return String.format("%s/%s", basePath, mutexWatershedDatasetFrom(threshold));
		}

		private static String mutexWatershedRelabeledDatasetFrom(final String basePath, final Double threshold) {
			return String.format("%s/%s", basePath, mutexWatershedRelabeledDatasetFrom(threshold));
		}

		private static String mutexWatershedMergedDatasetFrom(final String basePath, final Double threshold) {
			return String.format("%s/%s", basePath, mutexWatershedMergedDatasetFrom(threshold));
		}

		private static String mutexWatershedDatasetFrom(final Double threshold) {
			return (threshold == null)
					? "mutex-watershed"
					: String.format("mutex-watershed-threshold=%s", threshold);
		}

		private static String mutexWatershedRelabeledDatasetFrom(final Double threshold) {
			return String.format("%s-relabeled", mutexWatershedDatasetFrom(threshold));
		}

		private static String mutexWatershedMergedDatasetFrom(final Double threshold) {
			return String.format("%s-merged", mutexWatershedDatasetFrom(threshold));
		}
	}

	public static void main(String... args) throws IOException {

		final String containerPath = "/home/hanslovskyp/workspace/mutex-watershed/mutex-watershed-notebook/sample_A.n5";
		final String dataset = "affinities";
		final String outputDataset = "mutex-watershed";
		final String relabeledDataset = "mutex-watershed-relabeled";
		final String mergedDataset = "mutex-watershed-merged";
		final long[] outputSize = new long[] {256, 256, 256};//Intervals.dimensionsAsLongArray(Views.collapse(N5Utils.<FloatType>open(new N5FSReader(containerPath), dataset)));
		final N5WriterSupplier container = new N5WriterSupplier(containerPath);

		final SparkConf conf = new SparkConf().setAppName(SparkMutexWatersheds.class.getName());

		final long[][] offsets = {
				{-1, 0, 0}, {-2, 0, 0}, {-5, 0, 0}, {-10, 0, 0},
				{0, -1, 0}, {0, -2, 0}, {0, -5, 0}, {0, -10, 0},
				{0, 0, -1}, {0, 0, -2}, {0, 0, -5}, {0, 0, -10}};

		final ZeroExtendedSupplier<FloatType> affinitiesSupplier = new ZeroExtendedSupplier<>(container, dataset);

		try (final JavaSparkContext sc = new JavaSparkContext(conf)) {
			final StopWatch sw = StopWatch.createAndStart();
			runMutexWatersheds(
					sc,
					affinitiesSupplier,
					(Serializable & Supplier<RandomAccessible<UnsignedLongType>>) () -> ConstantUtils.constantRandomAccessible(new UnsignedLongType(1), 3),
					(Serializable & Supplier<RandomAccessible<FloatType>>) () -> ConstantUtils.constantRandomAccessible(new FloatType(0.0f), 3),
					container,
					outputDataset,
					relabeledDataset,
					mergedDataset,
					null,
					outputSize,
					new int[] {64, 64, 64},
					new int[] {1, 1, 1},
					new int[] {3, 3, 3},
					offsets);
			final double[] resolution = container.get().getAttribute(dataset, "resolution", double[].class);
			if (resolution != null) {
				container.get().setAttribute(outputDataset, "resolution", resolution);
				container.get().setAttribute(relabeledDataset, "resolution", resolution);
				container.get().setAttribute(mergedDataset, "resolution", resolution);
			}
			final double[] offset = container.get().getAttribute(dataset, "offset", double[].class);
			if (offset != null) {
				container.get().setAttribute(outputDataset, "offset", offset);
				container.get().setAttribute(relabeledDataset, "offset", offset);
				container.get().setAttribute(mergedDataset, "offset", offset);
			}
			sw.stop();
			System.out.println(String.format("Ran mutex watersheds in %f seconds", sw.seconds()));
		}

	}

	private static void runMutexWatersheds(
			final JavaSparkContext sc,
			final String container,
			final String affinitiesDataset,
			final String maskDataset,
			final String gliaDataset,
			final String mutexWatershedDataset,
			final String mutexWatershedRelabeledDataset,
			final String mutexWatershedMergedDataset,
			final Double threshold,
			final long[] outputSize,
			final int[] blockSize,
			final int[] blocksPerTask,
			final int[] superBlocksPerTask,
			final boolean overwriteExisting,
			final long[]... offsets) throws IOException {
		runMutexWatersheds(
				sc,
				container,
				container,
				container,
				container,
				affinitiesDataset,
				maskDataset,
				gliaDataset,
				mutexWatershedDataset,
				mutexWatershedRelabeledDataset,
				mutexWatershedMergedDataset,
				threshold,
				outputSize,
				blockSize,
				blocksPerTask,
				superBlocksPerTask,
				overwriteExisting,
				offsets);
	}

	private static <
			A extends NativeType<A> & RealType<A>,
			M extends NativeType<M> & IntegerType<M>,
			G extends NativeType<G> & RealType<G>> void runMutexWatersheds(
			final JavaSparkContext sc,
			final String affinitiesContainer,
			final String maskContainer,
			final String gliaContainer,
			final String outputContainer,
			final String affinitiesDataset,
			final String maskDataset,
			final String gliaDataset,
			final String mutexWatershedDataset,
			final String mutexWatershedRelabeledDataset,
			final String mutexWatershedMergedDataset,
			final Double threshold,
			final long[] outputSize,
			final int[] blockSize,
			final int[] blocksPerTask,
			final int[] superBlocksPerTask,
			final boolean overwriteExisting,
			final long[]... offsets) throws IOException {
		final N5WriterSupplier outputContainerSupplier = new N5WriterSupplier(outputContainer);
		if (outputContainerSupplier.get().datasetExists(mutexWatershedMergedDataset)
				&& Optional.ofNullable(outputContainerSupplier.get().getAttribute(mutexWatershedMergedDataset, "completedSuccessfully", Boolean.class)).orElse(false)
				&& !overwriteExisting) {
			System.out.println(String.format("Dataset %s already completed in %s: Skipping", mutexWatershedMergedDataset, outputContainer));
			return;
		}
		runMutexWatersheds(
				sc,
				new ZeroExtendedSupplier<A>(new N5ReaderSupplier(affinitiesContainer), affinitiesDataset),
				new ZeroExtendedSupplier<M>(new N5ReaderSupplier(maskContainer), maskDataset),
				new ZeroExtendedSupplier<G>(new N5ReaderSupplier(gliaContainer), gliaDataset),
				outputContainerSupplier,
				mutexWatershedDataset,
				mutexWatershedRelabeledDataset,
				mutexWatershedMergedDataset,
				threshold,
				outputSize,
				blockSize,
				blocksPerTask,
				superBlocksPerTask,
				offsets);
	}

	private static <
			A extends RealType<A> & NativeType<A>,
			M extends IntegerType<M>,
			G extends RealType<G>> void runMutexWatersheds(
			final JavaSparkContext sc,
			final Supplier<RandomAccessible<A>> affinitiesSupplier,
			final Supplier<RandomAccessible<M>> maskSupplier,
			final Supplier<RandomAccessible<G>> gliaSupplier,
			final Supplier<N5FSWriter> outputContainer,
			final String mutexWatershedDataset,
			final String mutexWatershedRelabeledDataset,
			final String mutexWatershedMergedDataset,
			final Double threshold,
			final long[] outputSize,
			final int[] blockSize,
			final int[] blocksPerTask,
			final int[] superBlocksPerTask,
			final long[]... offsets) throws IOException {

		outputContainer.get().createDataset(mutexWatershedDataset, outputSize, blockSize, DataType.UINT64, new GzipCompression());
		outputContainer.get().createDataset(mutexWatershedRelabeledDataset, outputSize, blockSize, DataType.UINT64, new GzipCompression());
		outputContainer.get().createDataset(mutexWatershedMergedDataset, outputSize, blockSize, DataType.UINT64, new GzipCompression());
		outputContainer.get().setAttribute(mutexWatershedMergedDataset, "completedSuccessfully", false);

		final double primitiveThreshold = threshold == null ? -1.0 : threshold;
		final boolean useThreshold = threshold != null && threshold >= 0.0;

		final int[] taskSize = new int[blockSize.length];
		Arrays.setAll(taskSize, d -> blockSize[d] * blocksPerTask[d]);

		final int[] superBlockSize = new int[blockSize.length];
		Arrays.setAll(superBlockSize, d -> taskSize[d] * superBlocksPerTask[d]);

		final List<IntervalWithOffset> tasksWithOffset = Grids
				.collectAllContainedIntervalsWithGridPositions(outputSize, taskSize)
				.stream()
				.map(p -> new IntervalWithOffset(p.getA()))
				.collect(Collectors.toList());

		final List<IntervalWithOffset> superBlocksWithOffset = Grids
				.collectAllContainedIntervalsWithGridPositions(outputSize, superBlockSize)
				.stream()
				.map(p -> new IntervalWithOffset(p.getA()))
				.collect(Collectors.toList());

		final int numChannels = offsets.length;

		final Map<IntervalWithOffset, Long> blocksWithCounts = sc
				.parallelize(superBlocksWithOffset)
				.map(bwo -> {
					final List<Interval> blocks = Grids.collectAllContainedIntervals(bwo.min, bwo.max, taskSize);
					final List<Tuple2<IntervalWithOffset, Long>> indices = new ArrayList<>();
					final RandomAccessible<A> affinityPredictions = affinitiesSupplier.get();
					final RandomAccessible<M> mask = maskSupplier.get();
					final RandomAccessible<G> glia = gliaSupplier.get();
					for (final Interval block : blocks) {
						final int numDims = block.numDimensions();
						final long[] min = Intervals.minAsLongArray(block);
						final long[] max = Intervals.maxAsLongArray(block);
						final long[] minWithChannels = new long[min.length + 1];
						final long[] maxWithChannels = new long[max.length + 1];
						System.arraycopy(min, 0, minWithChannels, 0, min.length);
						System.arraycopy(max, 0, maxWithChannels, 0, max.length);
						minWithChannels[minWithChannels.length - 1] = 0;
						maxWithChannels[maxWithChannels.length - 1] = numChannels - 1;
						final RandomAccessibleInterval<A> affinitiesInterval = Views.zeroMin(Views.interval(affinityPredictions, minWithChannels, maxWithChannels));
						final RandomAccessibleInterval<M> maskFrom = Views.interval(mask, block);
						final RandomAccessibleInterval<G> gliaFrom = Views.interval(glia, block);
						final StopWatch sw1 = StopWatch.createAndStart();
						final RandomAccessibleInterval<FloatType> affinitiesCopy = ArrayImgs.floats(Intervals.dimensionsAsLongArray(affinitiesInterval));
						for (int channel = 0; channel < numChannels; ++channel) {
							final long[] channelOffset = offsets[channel];
							final RandomAccessibleInterval<A> affinitiesSlice = Views.hyperSlice(affinitiesInterval, numDims, (long) channel);
							final RandomAccessibleInterval<FloatType> affinitiesCopySlice = Views.hyperSlice(affinitiesCopy, numDims, (long) channel);
							final RandomAccessibleInterval<M> maskTo = Views.interval(mask, Intervals.translate(block, channelOffset));
							final RandomAccessibleInterval<G> gliaTo = Views.interval(glia, Intervals.translate(block, channelOffset));

							LoopBuilder
									.setImages(affinitiesSlice, affinitiesCopySlice, maskFrom, maskTo, gliaFrom, gliaTo)
									.forEachPixel((s, t, mf, mt, gf, gt) -> {
										if (mf.getIntegerLong() != 1L || mt.getIntegerLong() != 1L)
											t.setReal(Double.NaN);
										else {
											double gwf = 1.0 - Math.min(Math.max(gf.getRealDouble(), 0.0), 1.0);
											double gwt = 1.0 - Math.min(Math.max(gt.getRealDouble(), 0.0), 1.0);
											t.setReal(s.getRealDouble() * gwf * gwt);
										}
									});

						}
						sw1.stop();
						System.out.println("Prepared affinities in " + StopWatch.secondsToString(sw1.seconds()));
//					return new Tuple2<>(bwo, affinitiesCopy);
//				})
//				.mapToPair(t -> {
						final RandomAccessibleInterval<FloatType> affinities = affinitiesCopy;
						final RandomAccessibleInterval<UnsignedLongType> target = ArrayImgs.unsignedLongs(Intervals.dimensionsAsLongArray(block));

						final double[] edgeProbabilities = Stream
								.of(offsets)
								.mapToDouble(o -> squaredSum(o) <= 1 ? 1.0:0.01)
								.toArray();

						final long seed = IntervalIndexer.positionToIndex(min, outputSize);
						final Random rng = new Random(seed);

						final StopWatch sw = StopWatch.createAndStart();
						if (useThreshold) {
							MutexWatershed.computeMutexWatershedClustering(
									affinities,
									target,
									offsets,
									edgeProbabilities,
									primitiveThreshold,
									rng::nextDouble);
						} else {
							MutexWatershed.computeMutexWatershedClustering(
									affinities,
									target,
									offsets,
									edgeProbabilities,
									attractiveEdges(offsets),
									rng::nextDouble);
						}
						sw.stop();
//						System.out.println("Ran mutex watersheds in " + StopWatch.secondsToString(sw.seconds()));

						final TLongLongHashMap counts = new TLongLongHashMap();
						Views.flatIterable(target).forEach(px -> {
							final long v = px.getIntegerLong();
							counts.put(v, counts.get(v) + 1);
						});


						long index = 1L;
						final TLongLongMap mapping = new TLongLongHashMap();
						mapping.put(0, 0);
						for (UnsignedLongType px : Views.flatIterable(target)) {
							final long k = px.getIntegerLong();
							if (counts.get(k) <= 1) {
								px.setZero();
								counts.remove(k);
								counts.put(0L, counts.get(0L) + 1);
							} else {
								if (!mapping.containsKey(k)) {
									mapping.put(k, index);
									++index;
								}
								px.setInteger(mapping.get(k));
							}
						}

						final long[] blockOffset = new long[min.length];
						Arrays.setAll(blockOffset, d -> min[d] / blockSize[d]);
						final DatasetAttributes attributes = new DatasetAttributes(outputSize, blockSize, DataType.UINT64, new GzipCompression());
						N5Utils.saveBlock(target, outputContainer.get(), mutexWatershedDataset, attributes, blockOffset);

						indices.add(new Tuple2<>(new IntervalWithOffset(block), index));

					}
					return indices;
				})
				.flatMap(List::iterator)
				.mapToPair(t -> t)
				.collectAsMap();

		long count = 0;
		final List<Tuple2<IntervalWithOffset, Long>> blocksWithIndexOffsets = new ArrayList<>();
		for (final IntervalWithOffset block : tasksWithOffset) {
			blocksWithIndexOffsets.add(new Tuple2<>(block, count + 1));
			count += blocksWithCounts.get(block);
		}

		sc
				.parallelizePairs(blocksWithIndexOffsets)
				.foreach(bwio -> {
					final IntervalWithOffset bwo = bwio._1();
					final Interval block = new FinalInterval(bwo.min, bwo.max);
					long index = bwio._2();
					final TLongLongHashMap mapping = new TLongLongHashMap();
					mapping.put(0, 0);
					final RandomAccessibleInterval<UnsignedLongType> labelData = N5Utils.open(outputContainer.get(), mutexWatershedDataset);
					final RandomAccessibleInterval<UnsignedLongType> relabeled = ArrayImgs.unsignedLongs(Intervals.dimensionsAsLongArray(block));
					final Cursor<UnsignedLongType> source = Views.flatIterable(Views.interval(labelData, block)).cursor();
					final Cursor<UnsignedLongType> target = Views.flatIterable(relabeled).cursor();
					while (source.hasNext()) {
						final long s = source.next().getIntegerLong();
						final UnsignedLongType t = target.next();
						if (!mapping.containsKey(s)) {
							mapping.put(s, index);
							++index;
						}
						t.setInteger(mapping.get(s));
					}

					final long[] blockOffset = new long[bwo.min.length];
					Arrays.setAll(blockOffset, d -> bwo.min[d] / blockSize[d]);
					final DatasetAttributes attributes = new DatasetAttributes(outputSize, blockSize, DataType.UINT64, new GzipCompression());
					N5Utils.saveBlock(relabeled, outputContainer.get(), mutexWatershedRelabeledDataset, attributes, blockOffset);
				});

		final List<Tuple2<long[], long[]>> mappings = sc
				.parallelize(tasksWithOffset)
				.map(bwo -> {
					final TLongList from = new TLongArrayList();
					final TLongList to = new TLongArrayList();

					final RandomAccessibleInterval<UnsignedLongType> labelData = N5Utils.open(outputContainer.get(), mutexWatershedRelabeledDataset);

					for (int d = 0; d < labelData.numDimensions(); ++d) {
						final TLongObjectMap<TLongLongMap> forwardCounts = new TLongObjectHashMap<>();
						final TLongObjectMap<TLongLongMap> backwardCounts = new TLongObjectHashMap<>();

						final long[] min = bwo.min.clone();
						final long[] max = bwo.max.clone();
						min[d] = max[d];
						if (max[d] == labelData.max(d))
							continue;

						final RandomAccessibleInterval<UnsignedLongType> slice1 = Views.interval(labelData, min, max);
						final RandomAccessibleInterval<UnsignedLongType> slice2 = Views.interval(labelData, Intervals.translate(slice1, 1L, d));
						LoopBuilder
								.setImages(slice1, slice2)
								.forEachPixel((s1, s2) -> {
									final long l1 = s1.getIntegerLong();
									final long l2 = s2.getIntegerLong();
									// l1 != l2 unless they are both 0 (or there is a bug)
									if (l1 != 0 && l2 != 0) {
										incrementFor(forwardCounts, l1, l2);
										incrementFor(backwardCounts, l2, l1);
									}
								});
						final TLongLongMap forwardArgmax = argMaxes(forwardCounts);
						final TLongLongMap backwardArgmax = argMaxes(backwardCounts);

						forwardArgmax.forEachEntry((key, argMax) -> {
							if (backwardArgmax.get(argMax) == key) {
								from.add(key);
								to.add(argMax);
							}
							return true;
						});
					}

					return new Tuple2<>(from.toArray(), to.toArray());
				})
				.collect();

		final TLongLongHashMap parents = new TLongLongHashMap();
		final UnionFind uf = new LongHashMapUnionFind(parents, 0, Long::compare);

		mappings.forEach(p -> {
			final long[] keys = p._1();
			final long[] values = p._2();
			for (int i = 0; i < keys.length; ++i) {
				uf.join(uf.findRoot(keys[i]), uf.findRoot(values[i]));
			}
		});

		parents.forEachKey(key -> {
			uf.findRoot(key);
			return true;
		});

		final long[] mappingKeys = parents.keys();
		final long[] mappingValues = parents.values();

		sc
				.parallelize(tasksWithOffset)
				.foreach(bwo -> {
					final Interval block = new FinalInterval(bwo.min, bwo.max);
					final TLongLongMap mapping = new TLongLongHashMap(mappingKeys, mappingValues);
					final RandomAccessibleInterval<UnsignedLongType> labelData = N5Utils.open(outputContainer.get(), mutexWatershedRelabeledDataset);
					final RandomAccessibleInterval<UnsignedLongType> remapped = ArrayImgs.unsignedLongs(Intervals.dimensionsAsLongArray(block));

					LoopBuilder
							.setImages(Views.interval(labelData, block), remapped)
							.forEachPixel((s, t) -> {
								final long k = s.getIntegerLong();
								t.setInteger(mapping.containsKey(k) ? mapping.get(k) : k);
							});


					final long[] blockOffset = new long[bwo.min.length];
					Arrays.setAll(blockOffset, d -> bwo.min[d] / blockSize[d]);
					final DatasetAttributes attributes = new DatasetAttributes(outputSize, blockSize, DataType.UINT64, new GzipCompression());
					N5Utils.saveBlock(remapped, outputContainer.get(), mutexWatershedMergedDataset, attributes, blockOffset);

				});

		// Success!!
		outputContainer.get().setAttribute(mutexWatershedMergedDataset, "completedSuccessfully", true);


	}

	private static TLongLongMap argMaxes(final TLongObjectMap<TLongLongMap> counts) {
		final TLongLongMap argMaxes = new TLongLongHashMap();
		counts.forEachEntry((key, value) -> {
			long argMax = -1;
			long maxCount = Long.MIN_VALUE;
			final TLongLongIterator iterator = value.iterator();
			while (iterator.hasNext()) {
				iterator.advance();
				final long count = iterator.value();
				if (count > maxCount) {
					maxCount = count;
					argMax = iterator.key();
				}
			}
			if (maxCount > 0)
				argMaxes.put(key, argMax);
			return true;
		});
		return argMaxes;
	}

	private static void incrementFor(final TLongObjectMap<TLongLongMap> counts, final long from, final long to) {
		if (!counts.containsKey(from))
			counts.put(from, new TLongLongHashMap());
		final TLongLongMap toCounts = counts.get(from);
		toCounts.put(to, toCounts.get(to) + 1);
	}

	private static long squaredSum(long... values) {
		long sum = 0;
		for (long v : values)
			sum += v * v;
		return sum;
	}

	private static boolean[] attractiveEdges(long[]... offsets) {
		final boolean[] attractiveEdges = new boolean[offsets.length];
		for (int d = 0; d < offsets.length; ++d)
			attractiveEdges[d] = squaredSum(offsets[d]) <= 1L;
		return attractiveEdges;
	}

	private interface SerializableSupplier<T> extends Supplier<T>, Serializable {

	}

	private static class N5WriterSupplier implements SerializableSupplier<N5FSWriter> {

		private final String containerPath;

		private N5WriterSupplier(final String containerPath) {
			this.containerPath = containerPath;
		}

		@Override
		public N5FSWriter get() {
			try {
				return new N5FSWriter(containerPath);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
	}

	private static class N5ReaderSupplier implements SerializableSupplier<N5FSReader> {

		private final String containerPath;

		private N5ReaderSupplier(final String containerPath) {
			this.containerPath = containerPath;
		}

		@Override
		public N5FSReader get() {
			try {
				return new N5FSReader(containerPath);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
	}

	private static class ZeroExtendedSupplier<T extends NativeType<T> & NumericType<T>> implements SerializableSupplier<RandomAccessible<T>> {
		private final SerializableSupplier<? extends N5Reader> container;

		private final String dataset;

		private ZeroExtendedSupplier(
				final SerializableSupplier<? extends N5Reader> container,
				final String dataset) {
			this.container = container;
			this.dataset = dataset;
		}

		@Override
		public RandomAccessible<T> get() {
			try {
				final N5Reader container = this.container.get();
				final double[] offset = container.getAttribute(dataset, "offset", double[].class);
				if (offset == null || Arrays.stream(offset).allMatch(d -> d ==0.0))
					return Views.extendZero(N5Utils.<T>open(container, dataset));
				final double[] resolution = Optional.ofNullable(container.getAttribute(dataset, "resolution", double[].class)).orElse(new double[] {1.0, 1.0, 1.0});
				final long[] offsetInVoxels = new long[offset.length];
				Arrays.setAll(offsetInVoxels, d -> (long) (offset[d] / resolution[d]));
				return Views.extendZero(Views.translate(N5Utils.<T>open(container, dataset), offsetInVoxels));
			} catch (final IOException e) {
				throw new RuntimeException(e);
			}
		}
	}

	private static class IntervalWithOffset implements Serializable {

		public final long[] min;

		public final long[] max;

		public IntervalWithOffset(long[] min, long[] max) {
			this.min = min;
			this.max = max;
		}

		public IntervalWithOffset(final Interval interval) {
			this(Intervals.minAsLongArray(interval), Intervals.maxAsLongArray(interval));
		}

		@Override
		public int hashCode() {
			return Arrays.hashCode(this.min);
		}

		@Override
		public boolean equals(final Object other) {
			return other instanceof IntervalWithOffset && Arrays.equals(min, ((IntervalWithOffset)other).min);
		}
	}

	private static <T extends NumericType<T>> boolean anyZero(final RandomAccessibleInterval<T> rai) {
		final T zero = Util.getTypeFromInterval(rai);
		zero.setZero();
		return anyMatches(Views.iterable(rai), zero);
	}

	private static <T extends NumericType<T>> boolean anyMatches(final Iterable<T> iterable, final T comp) {
		for (final T t : iterable)
			if (t.valueEquals(comp))
				return true;
		return false;
	}

}
