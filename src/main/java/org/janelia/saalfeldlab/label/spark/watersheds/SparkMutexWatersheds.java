package org.janelia.saalfeldlab.label.spark.watersheds;

import gnu.trove.map.TLongLongMap;
import gnu.trove.map.hash.TLongLongHashMap;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.ConstantUtils;
import net.imglib2.util.IntervalIndexer;
import net.imglib2.util.Intervals;
import net.imglib2.util.StopWatch;
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
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SparkMutexWatersheds {

	public static void main(String... args) throws IOException {

		final String containerPath = "/home/hanslovskyp/workspace/mutex-watershed/mutex-watershed-notebook/sample_A.n5";
		final String dataset = "affinities";
		final String outputDataset = "mutex-watershed";
		final long[] outputSize = Intervals.dimensionsAsLongArray(Views.collapse(N5Utils.<FloatType>open(new N5FSReader(containerPath), dataset)));
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
					outputSize,
					new int[] {64, 64, 64},
					new int[] {1, 1, 1},
					offsets);
			new N5FSWriter(containerPath).setAttribute(outputDataset, "resolution", new N5FSWriter(containerPath).getAttribute(dataset, "resolution", double[].class));
			new N5FSWriter(containerPath).setAttribute(outputDataset, "offset", new N5FSWriter(containerPath).getAttribute(dataset, "offset", double[].class));
			sw.stop();
			System.out.println(String.format("Ran mutex watersheds in %f seconds", sw.seconds()));
		}

	}

	private static <
			A extends RealType<A>,
			M extends IntegerType<M>,
			G extends RealType<G>> void runMutexWatersheds(
			final JavaSparkContext sc,
			final Supplier<RandomAccessible<A>> affinitiesSupplier,
			final Supplier<RandomAccessible<M>> maskSupplier,
			final Supplier<RandomAccessible<G>> gliaSupplier,
			final Supplier<N5FSWriter> outputContainer,
			final String outputDataset,
			final long[] outputSize,
			final int[] blockSize,
			final int[] blocksPerTask,
			final long[]... offsets) throws IOException {

		outputContainer.get().createDataset(outputDataset, outputSize, blockSize, DataType.UINT64, new GzipCompression());

		final int[] taskSize = new int[blockSize.length];
		Arrays.setAll(taskSize, d -> blockSize[d] * blocksPerTask[d]);

		final List<IntervalWithOffset> blocksWithOffsets = Grids
				.collectAllContainedIntervalsWithGridPositions(outputSize, taskSize)
				.stream()
				.map(p -> new IntervalWithOffset(p.getA(), p.getB()))
				.collect(Collectors.toList());

		final int numChannels = offsets.length;

		final Map<IntervalWithOffset, Long> blocksWithCounts = sc
				.parallelize(blocksWithOffsets)
				.mapToPair(bwo -> {
					final Interval block = new FinalInterval(bwo.min, bwo.max);
					final int numDims = block.numDimensions();
					final long[] minWithChannels = new long[bwo.min.length + 1];
					final long[] maxWithChannels = new long[bwo.max.length + 1];
					System.arraycopy(bwo.min, 0, minWithChannels, 0, bwo.min.length);
					System.arraycopy(bwo.max, 0, maxWithChannels, 0, bwo.max.length);
					minWithChannels[minWithChannels.length - 1] = 0;
					maxWithChannels[maxWithChannels.length - 1] = numChannels - 1;
					final RandomAccessibleInterval<A> affinitiesInterval = Views.interval(affinitiesSupplier.get(), minWithChannels, maxWithChannels);
					final RandomAccessibleInterval<DoubleType> affinitiesCopy = ArrayImgs.doubles(Intervals.dimensionsAsLongArray(affinitiesInterval));
					final RandomAccessible<M> mask = maskSupplier.get();
					final RandomAccessible<G> glia = gliaSupplier.get();
					final RandomAccessibleInterval<M> maskFrom = Views.interval(mask, block);
					final RandomAccessibleInterval<G> gliaFrom = Views.interval(glia, block);
					for (int channel = 0; channel < numChannels; ++channel) {
						final long[] channelOffset = offsets[channel];
						final RandomAccessibleInterval<A> affinitiesSlice = Views.hyperSlice(affinitiesInterval, numDims, (long) channel);
						final RandomAccessibleInterval<DoubleType> affinitiesCopySlice = Views.hyperSlice(affinitiesCopy, numDims, (long) channel);
						final RandomAccessibleInterval<M> maskTo = Views.interval(mask, Intervals.translate(block, channelOffset));
						final RandomAccessibleInterval<G> gliaTo = Views.interval(glia, Intervals.translate(block, channelOffset));

						LoopBuilder
								.setImages(affinitiesSlice, affinitiesCopySlice, maskFrom, maskTo, gliaFrom, gliaTo)
								.forEachPixel((s, t, mf, mt, gf, gt) -> {
									if (mf.getIntegerLong() != 1L || mt.getIntegerLong() != 1L)
										t.set(Double.NaN);
									else {
										double gwf = 1.0 - Math.min(Math.max(gf.getRealDouble(), 0.0), 1.0);
										double gwt = 1.0 - Math.min(Math.max(gt.getRealDouble(), 0.0), 1.0);
										t.setReal(Math.min(Math.max(s.getRealDouble(), 0.0), 1.0) * gwf * gwt);
									}
								});

					}
					return new Tuple2<>(bwo, affinitiesCopy);
				})
				.mapToPair(t -> {
					final IntervalWithOffset bwo = t._1();
					final Interval block = new FinalInterval(bwo.min, bwo.max);
					final RandomAccessibleInterval<DoubleType> affinities = t._2();
					final RandomAccessibleInterval<UnsignedLongType> target = ArrayImgs.unsignedLongs(Intervals.dimensionsAsLongArray(block));

					final double[] edgeProbabilities = Stream
							.of(offsets)
							.mapToDouble(o -> squaredSum(o) <= 1 ? 1.0 : 0.05)
							.toArray();

					final long seed = IntervalIndexer.positionToIndex(bwo.min, outputSize);
					final Random rng = new Random(seed);

					MutexWatershed.computeMutexWatershedClustering(
							affinities,
							target,
							offsets,
							edgeProbabilities,
							attractiveEdges(offsets),
							rng::nextDouble);

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
					final long[] blockOffset = new long[bwo.min.length];
					Arrays.setAll(blockOffset, d -> bwo.min[d] / blockSize[d]);
					final DatasetAttributes attributes = new DatasetAttributes(outputSize, blockSize, DataType.UINT64, new GzipCompression());
					N5Utils.saveBlock(target, outputContainer.get(), outputDataset, attributes, blockOffset);

					return new Tuple2<IntervalWithOffset, Long>(bwo, index);
				})
				.collectAsMap();


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
				return Views.extendZero(N5Utils.<T>open(container.get(), dataset));
			} catch (final IOException e) {
				throw new RuntimeException(e);
			}
		}
	}

	private static class IntervalWithOffset implements Serializable {

		public final long[] min;

		public final long[] max;

		public final long[] pos;

		public IntervalWithOffset(long[] min, long[] max, long[] pos) {
			this.min = min;
			this.max = max;
			this.pos = pos;
		}

		public IntervalWithOffset(final Interval interval, long[] pos) {
			this(Intervals.minAsLongArray(interval), Intervals.maxAsLongArray(interval), pos);
		}

		@Override
		public int hashCode() {
			return Arrays.hashCode(this.pos);
		}

		@Override
		public boolean equals(final Object other) {
			return other instanceof IntervalWithOffset && Arrays.equals(pos, ((IntervalWithOffset)other).pos);
		}
	}

//	private static final String RESOLUTION_KEY = "resolution";
//
//	private static final String OFFSET_KEY = "offset";
//
//	private static final String ARGUMENTS_KEY = "arguments";
//
//	private static final String ARGV_KEY = "argumentVector";
//
//	private static final String LABEL_UTILITIES_SPARK_KEY = "label-utilities-spark";
//
//	private static final String VERSION_KEY = "version";
//
//	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());
//
//	private static class Args implements Serializable, Callable<Integer> {
//
//		@Expose
//		@CommandLine.Option(names = "--input-container", paramLabel = "INPUT_CONTAINER", description = "Path to N5 container with affinities dataset.", required = true)
//		String inputContainer = null;
//
//		@Expose
//		@CommandLine.Option(names = "--output-container", paramLabel = "OUTPUT_CONTAINER", description = "Path to output container. Defaults to INPUT_CONTAINER.")
//		String outputContainer = null;
//
//		@Expose
//		@CommandLine.Option(names = "--offsets", paramLabel = "OFFSETS", arity = "1..*", split = ",")
//		long[][] offsets = {
//				{-1, 0, 0}, {-2, 0, 0}, {-5, 0, 0}, {-10, 0, 0},
//				{0, -1, 0}, {0, -2, 0}, {0, -5, 0}, {0, -10, 0},
//				{0, 0, -1}, {0, 0, -2}, {0, 0, -5}, {0, 0, -10},
//		};
//
//		@Expose
//		@CommandLine.Option(names = "--mask-container", paramLabel = "MASK_CONTAINER", description = "Path to output container. Defaults to INPUT_CONTAINER.")
//		String maskContainer = null;
//
//		@Expose
//		@CommandLine.Option(names = "--affinities-dataset", paramLabel = "AFFINITIES", description = "Path of affinities dataset in INPUT_CONTAINER.", required = true)
//		String affinities;
//
//		@Expose
//		@CommandLine.Option(names = "--mask-dataset", paramLabel = "MASK", description = "Path of mask dataset in MASK_CONTAINER.")
//		String mask;
//
//		@Expose
//		@CommandLine.Option(names = "--glia-dataset", paramLabel = "GLIA_DATASET", description = "Weigh affinities by glia prediction (if provided)")
//		String gliaPrediction = null;
//
//		@Expose
//		@CommandLine.Option(names = "--label-datasets-prefix", paramLabel = "LABEL_DATASETS_PREFIX", defaultValue = "volumes/labels", description = "Will be prepended to datasets: ${LABEL_DATASETS_PREFIX}/${LABEL_DATASET}. Set to empty String to ignore.")
//		String labelDatasetsPrefix;
//
//		@Expose
//		@CommandLine.Option(names = "--mutex-watersheds-dataset", paramLabel = "WATERSHEDS", description = "Path in container: ${LABEL_DATASETS_PREFIX}/${WATERSHEDS}", defaultValue = "mutex-watersheds")
//		String mutexWatersheds;
//
//		@Expose
//		@CommandLine.Option(names = "--block-merged-dataset", paramLabel = "MERGED", description = "Path in container: ${LABEL_DATASETS_PREFIX}/${MERGED}", defaultValue = "mutex-watersheds-merged")
//		String blockMerged;
//
//		@Expose
//		@CommandLine.Option(names = "--block-size", paramLabel = "BLOCK_SIZE", description = "Block size of output.", split = ",", defaultValue = "64,64,64")
//		int[] blockSize;
//
//		@Expose
//		@CommandLine.Option(names = "--blocks-per-task", paramLabel = "BLOCKS_PER_TASK", description = "How many blocks to combine for watersheds/connected components (one value per dimension)", split = ",", defaultValue = "1,1,1")
//		int[] blocksPerTask;
//
//		@Expose
//		@CommandLine.Option(names = "--threshold", paramLabel = "THRESHOLD", description = "Threshold for affinities. If no threshold is provided, all nearest neighbor affinities will be used for attractive edges, all other affinities will be repulsive.")
//		Double threshold;
//
//		@Expose
//		@CommandLine.Option(names = "--halo", paramLabel = "HALO", description = "Include halo region to run connected components/watersheds", split = ",")
//		int[] halo = {0, 0, 0};
//
//		@Expose
//		@CommandLine.Option(names = "--cropped-datasets-pattern", paramLabel = "CROPPED_DATASETS_PATTERN", description = "All data sets are stored with halo by default. Cropped versions according to this.", defaultValue = "%s-cropped")
//		String croppedDatasetPattern;
//
//		@Expose
//		@CommandLine.Option(names = "--relabel", paramLabel = "RELABEL", description = "Relabel all label data sets for unique labels", defaultValue = "false")
//		Boolean relabel;
//
//		@Expose
//		@CommandLine.Option(names = "--revert-array-attributes", paramLabel = "REVERT", description = "Revert all array attributes (that are not dataset attributes)", defaultValue = "false")
//		Boolean revertArrayAttributes;
//
//		@CommandLine.Option(names = "--json-pretty-print", defaultValue = "true")
//		transient Boolean prettyPrint;
//
//		@CommandLine.Option(names = "--json-disable-html-escape", defaultValue = "true")
//		transient Boolean disbaleHtmlEscape;
//
//		@CommandLine.Option(names = {"-h", "--help"}, usageHelp = true, description = "Display this help and exit")
//		private Boolean help;
//
//		@CommandLine.Option(names = "--min-object-size", description = "Only objects larger than `MIN_OBJECT_SIZE' will be foreground, if specified")
//		private Long minObjectSize;
//
//		@Override
//		public Integer call() throws IOException {
//
//			if (help != null && help)
//				return 0;
//
//			mutexWatersheds = String.join("/", labelDatasetsPrefix, mutexWatersheds);
//			blockMerged = String.join("/", labelDatasetsPrefix, blockMerged);
//
//			final N5WriterSupplier n5in = new N5WriterSupplier(inputContainer, prettyPrint, disbaleHtmlEscape);
//
//			final N5WriterSupplier maskIn = maskContainer==null
//					? n5in
//					:new N5WriterSupplier(maskContainer, prettyPrint, disbaleHtmlEscape);
//
//			final N5WriterSupplier n5out = outputContainer==null
//					? n5in
//					:new N5WriterSupplier(outputContainer, prettyPrint, disbaleHtmlEscape);
//
//			final DatasetAttributes inputAttributes = n5in.get().getDatasetAttributes(affinities);
//			final long[] inputDims = inputAttributes.getDimensions();
//			final long[] outputDims = new long[inputDims.length - 1];
//			Arrays.setAll(outputDims, d -> inputDims[d]);
//
//			final Map<String, Object> labelUtilitiesSparkAttributes = new HashMap<>();
//			labelUtilitiesSparkAttributes.put(ARGUMENTS_KEY, this);
//			// TODO how to store argv?
////			labelUtilitiesSparkAttributes.put(ARGV_KEY, argv);
//			labelUtilitiesSparkAttributes.put(VERSION_KEY, Version.VERSION_STRING);
//			final Map<String, Object> attributes = with(new HashMap<>(), LABEL_UTILITIES_SPARK_KEY, labelUtilitiesSparkAttributes);
//
//
//			final int[] taskBlockSize = IntStream.range(0, blockSize.length).map(d -> blockSize[d] * blocksPerTask[d]).toArray();
//			final boolean hasHalo = Arrays.stream(halo).filter(h -> h!=0).count() > 0;
//			if (hasHalo)
//				throw new UnsupportedOperationException("Halo currently not supported, please omit halo option!");
//
//			String[] uint64Datasets = {mutexWatersheds, blockMerged};
//			String[] uint8Datasets = {};
//
//			final double[] resolution = reverted(Optional.ofNullable(n5in.get().getAttribute(affinities, RESOLUTION_KEY, double[].class)).orElse(ones(outputDims.length)), revertArrayAttributes);
//			final double[] offset = reverted(Optional.ofNullable(n5in.get().getAttribute(affinities, OFFSET_KEY, double[].class)).orElse(new double[outputDims.length]), revertArrayAttributes);
//			attributes.put(RESOLUTION_KEY, resolution);
//			attributes.put(OFFSET_KEY, offset);
//
//			final Map<String, DatasetAttributes> datasets = new HashMap<>();
//			Arrays.asList(uint64Datasets).forEach(ds -> datasets.put(ds, new DatasetAttributes(outputDims, blockSize, DataType.UINT64, new GzipCompression())));
//			Arrays.asList(uint8Datasets).forEach(ds -> datasets.put(ds, new DatasetAttributes(outputDims, blockSize, DataType.UINT8, new GzipCompression())));
//
//			if (hasHalo) {
//				prepareOutputDatasets(
//						n5out.get(),
//						datasets.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, e -> new DatasetAttributes(e.getValue().getDimensions(), taskBlockSize, e.getValue().getDataType(), e.getValue().getCompression()))),
//						attributes);
//			}
//
//			prepareOutputDatasets(
//					n5out.get(),
//					datasets.entrySet().stream().collect(Collectors.toMap(entry -> hasHalo ? String.format(croppedDatasetPattern, entry.getKey()):entry.getKey(), Map.Entry::getValue)),
//					attributes);
//
//			final SparkConf conf = new SparkConf().setAppName(MethodHandles.lookup().lookupClass().getName());
//			try (final JavaSparkContext sc = new JavaSparkContext(conf)) {
//
//				LOG.info("Input  dims:   {}", inputDims);
//				LOG.info("Output dims:   {}", outputDims);
//				run(
//						sc,
//						new CropAffinities(n5in, maskIn, n5in, affinities, mask, gliaPrediction, offsets, IntStream.of(halo).mapToLong(i -> i).toArray()),
//						offsets,
//						n5out,
//						outputDims,
//						IntStream.of(halo).mapToLong(i -> i).toArray(),
////					new SerializableMergeWatershedsMinThresholdSupplier(threshold),
//						null, //new SerializableMergeWatershedsMedianThresholdSupplier(threshold),
//						blockMerged,
//						mutexWatersheds,
//						croppedDatasetPattern,
//						blockSize,
//						blocksPerTask,
//						relabel);
//			}
//
//			return 0;
//		}
//	}
//
//	public static void main(final String[] argv) throws IOException {
//
//		run(argv);
//
//	}
//
//	public static void run(final String... argv) throws IOException {
//
//		final Args args = new Args();
//		CommandLine.call(args, argv);
//	}
//
//	public static void run(
//			final JavaSparkContext sc,
//			final PairFunction<Interval, Interval, RandomAccessibleInterval<FloatType>> affinitiesProvider,
//			final long[][] offsets,
//			final Supplier<? extends N5Writer> n5out,
//			final long[] outputDims,
//			final long[] halo,
//			final Supplier<MergeWatersheds> mergeWatershedregions,
//			final String blockMerged,
//			final String watersheds,
//			final String croppedDatasetPattern,
//			final int[] blockSize,
//			final int[] blocksPerTask,
//			final boolean relabel) throws IOException {
//
//		final boolean hasHalo = Arrays.stream(halo).filter(h -> h!=0).count() > 0;
//		final int[] watershedBlockSize = IntStream.range(0, blockSize.length).map(d -> blockSize[d] * blocksPerTask[d]).toArray();
//		final List<Tuple2<long[], long[]>> watershedBlocks = Grids
//				.collectAllContainedIntervals(outputDims, watershedBlockSize)
//				.stream()
//				.map(i -> new Tuple2<>(Intervals.minAsLongArray(i), Intervals.maxAsLongArray(i)))
//				.collect(Collectors.toList());
//		;
//		final long[] negativeHalo = new long[halo.length];
//		Arrays.setAll(negativeHalo, d -> -halo[d]);
//
//		final List<Tuple2<Tuple2<long[], long[]>, Integer>> idCounts = sc
//				.parallelize(watershedBlocks)
//				.map(t -> (Interval) new FinalInterval(t._1(), t._2()))
//				.mapToPair(affinitiesProvider)
//				.mapToPair(t -> {
//					final Interval block = t._1();
//					final RandomAccessibleInterval<FloatType> affinities = t._2();
//					final CellGrid grid = new CellGrid(outputDims, blockSize);
//					final CellGrid watershedsGrid = new CellGrid(outputDims, watershedBlockSize);
//					LOG.debug("Got grids {} and {}", grid, watershedsGrid);
//
//					final long[] blockOffset = Intervals.minAsLongArray(block);
//					final long[] watershedsBlockOffset = blockOffset.clone();
//					grid.getCellPosition(blockOffset, blockOffset);
//					watershedsGrid.getCellPosition(watershedsBlockOffset, watershedsBlockOffset);
//					LOG.debug("min={} blockOffset={} watershedsBlockOffset={}", Intervals.minAsLongArray(block), blockOffset, watershedsBlockOffset);
//
//					final long[] dims = Intervals.dimensionsAsLongArray(Views.hyperSlice(affinities, affinities.numDimensions() - 1, affinities.min(affinities.numDimensions() - 1)));
//					final ArrayImg<UnsignedLongType, LongArray> labels = ArrayImgs.unsignedLongs(dims);
//
//
//					final Interval relevantInterval = Intervals.expand(labels, negativeHalo);
//
//					final DatasetAttributes croppedAttributes = new DatasetAttributes(outputDims, blockSize, DataType.UINT64, new GzipCompression());
//					final DatasetAttributes watershedAttributes = new DatasetAttributes(outputDims, watershedBlockSize, DataType.UINT64, new GzipCompression());
//					final long nextId = MutexWatershed.computeMutexWatershedClustering(
//							affinities,
//							labels,
//							offsets,
//							Arrays.stream(offsets).mapToDouble(o -> isNearestNeighbor(o) ? 1.0:0.05).toArray(),
//							0.5);
//
//					// TODO find better logic instead of label == 0 check
//					final TLongSet ids = new TLongHashSet();
//					final TLongIntHashMap counts = new TLongIntHashMap();
//					for (final UnsignedLongType pix : labels) {
//						final long label = pix.getIntegerLong();
//						if (label==0)
//							counts.put(label, counts.get(label) + 1);
//						else {
//							ids.add(label);
//							counts.put(label, counts.get(label) + 1);
//						}
//					}
//
//					// TODO filter by size!!
////					for (final UnsignedLongType pix : labels) {
////						final long label = pix.getIntegerLong();
////						if (counts.get(label)==1) {
////							counts.remove(label);
////							ids.remove(label);
////							counts.put(0, counts.get(0) + 1);
////							pix.setZero();
////						}
////					}
////
//					N5Utils.saveBlock(Views.interval(labels, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, watersheds):watersheds, croppedAttributes, blockOffset);
//					if (hasHalo) {
//						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(labels), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
//						n5out.get().writeBlock(watersheds, watershedAttributes, dataBlock);
//					}
//
//					return new Tuple2<>(new Tuple2<>(Intervals.minAsLongArray(t._1()), Intervals.maxAsLongArray(t._1())), ids.size());
//				})
//				.collect();
//
//		long startIndex = 1;
//		final List<Tuple2<Tuple2<long[], long[]>, Long>> idOffsets = new ArrayList<>();
//		for (final Tuple2<Tuple2<long[], long[]>, Integer> idCount : idCounts) {
//			idOffsets.add(new Tuple2<>(idCount._1(), startIndex));
//			startIndex += idCount._2();
//		}
//
//		// TODO: Relabel!!
////		if (relabel) {
////			final long numBlocks = sc
////					.parallelizePairs(idOffsets)
////					.map(t -> {
////						LOG.debug("Relabeling block ({} {}) starting at id {}", t._1()._1(), t._1()._2(), t._2());
////						final N5Writer n5 = n5out.get();
////						final Interval interval = new FinalInterval(t._1()._1(), t._1()._2());
//////						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, interval, t._2());
////						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, merged) : merged, merged, interval, t._2());
//////						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, interval, t._2());
////						if (hasHalo)
////							throw new UnsupportedOperationException("Halo relabeling not implemented yet!");
////
////						// TODO do halo relabeling
////
////						return true;
////					})
////					.count();
////			LOG.debug("Relabeled {} blocks", numBlocks);
////			final long maxId = startIndex;
////			LOG.info("Found max id {}", maxId);
////			n5out.get().setAttribute(hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, "maxId", maxId);
////			n5out.get().setAttribute(hasHalo ? String.format(croppedDatasetPattern, merged) : merged, "maxId", maxId);
////			n5out.get().setAttribute(hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, "maxId", maxId);
////
////			if (hasHalo)
////				throw new UnsupportedOperationException("Halo relabeling not implemented yet!");
////
////			if (maxId + 2 > Integer.MAX_VALUE)
////				throw new RuntimeException("Currently only Integer.MAX_VALUE labels supported");
////
//////			final IntArrayUnionFind uf = findOverlappingLabelsArgMaxNoHalo(sc, n5out, merged, new IntArrayUnionFind((int) (maxId + 2)), outputDims, blockSize, blocksPerTask, 0);
////			final LongHashMapUnionFind uf = findOverlappingLabelsThresholdMedianEdgeAffinities(
////					sc,
////					n5out,
////					n5in,
////					merged,
////					averagedAffinities,
////					mergeFragmentsThreshold,
////					new LongHashMapUnionFind(),
////					outputDims,
////					blockSize,
////					blocksPerTask,
////					0);
////			LOG.debug("{} sets are grouped into {} sets", uf.size(), uf.setCount());
////
////			final List<Tuple2<Tuple2<long[], long[]>, Tuple2<long[], long[]>>> finalMappings = new ArrayList<>();
////			long maxRoot = 0;
////			for (int index = 0; index < idOffsets.size(); ++index) {
////				final Tuple2<Tuple2<long[], long[]>, Long> idOffset = idOffsets.get(index);
////				final long minLabel = idOffset._2();
////				final long maxLabel = index < idOffsets.size() - 1
////						? idOffsets.get(index + 1)._2()
////						: (maxId);
////
////				LOG.debug("Max label = {} min label = {} for block ({} {})", maxLabel, minLabel, idOffset._1()._1(), idOffset._1()._2());
////				final long[] keys = new long[(int) (maxLabel - minLabel)];
////				final long[] vals = new long[keys.length];
////
////
////				for (int i = 0; i < keys.length; ++i) {
////					final long k = i + minLabel;
////					final long root = uf.findRoot(k);
////					keys[i] = k;
////					vals[i] = root;
////					if (root > maxRoot) {
////						maxRoot = root;
////					}
////				}
////				finalMappings.add(new Tuple2<>(idOffset._1(), new Tuple2<>(keys, vals)));
////			}
////			n5out.get().setAttribute(blockMerged, "maxId", maxRoot);
////
////			sc
////					.parallelize(finalMappings)
////					.foreach(t -> {
////						final Interval interval = new FinalInterval(t._1()._1(), t._1()._2());
////						final TLongLongMap mapping = new TLongLongHashMap(t._2()._1(), t._2()._2());
////
//////						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, interval, mapping);
////						relabel(n5out.get(), hasHalo ? String.format(croppedDatasetPattern, merged) : merged, blockMerged, interval, mapping);
//////						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, interval, mapping);
////						if (hasHalo)
////							throw new UnsupportedOperationException("Halo relabeling not implemented yet!");
////					});
////
////		}
//	}
//
//	private static void relabel(
//			final N5Writer n5,
//			final String source,
//			final String target,
//			final Interval interval,
//			final TLongLongMap mapping) throws IOException {
//		SparkMutexWatersheds.<UnsignedLongType>relabel(n5, source, target, interval, (src, tgt) -> {
//			final long val = mapping.get(src.getIntegerLong());
//			if (val != 0)
//				tgt.set(val);
//		});
//	}
//
//	private static void relabel(
//			final N5Writer n5,
//			final String source,
//			final String target,
//			final Interval interval,
//			final long addIfNotZero) throws IOException {
//		final CachedMapper mapper = new CachedMapper(addIfNotZero);
//		SparkMutexWatersheds.<UnsignedLongType>relabel(n5, source, target, interval, (src, tgt) -> tgt.set(mapper.applyAsLong(src.getIntegerLong())));
//	}
//
//	private static <T extends IntegerType<T> & NativeType<T>> void relabel(
//			final N5Writer n5,
//			final String source,
//			final String target,
//			final Interval interval,
//			final BiConsumer<T, T> idMapping) throws IOException {
//		final DatasetAttributes attributes = n5.getDatasetAttributes(source);
//		final CellGrid grid = new CellGrid(attributes.getDimensions(), attributes.getBlockSize());
//		final RandomAccessibleInterval<T> data = Views.interval(N5Utils.<T>open(n5, source), interval);
//		final RandomAccessibleInterval<T> copy = new ArrayImgFactory<>(Util.getTypeFromInterval(data).createVariable()).create(data);
//		LOG.debug("Input size {} -- output size {}", Intervals.dimensionsAsLongArray(data), Intervals.dimensionsAsLongArray(copy));
//		long maxId = Long.MIN_VALUE;
//		{
//			final Cursor<T> src = Views.flatIterable(data).cursor();
//			final Cursor<T> tgt = Views.flatIterable(copy).cursor();
//			while (src.hasNext()) {
//				idMapping.accept(src.next(), tgt.next());
//				maxId = Math.max(tgt.get().getIntegerLong(), maxId);
//			}
//			LOG.debug("Remapped to max id {}", maxId);
//		}
//		final long[] blockPos = Intervals.minAsLongArray(interval);
//		grid.getCellPosition(blockPos, blockPos);
//		N5Utils.saveBlock(copy, n5, target, attributes, blockPos);
//	}
//
//	private static void relabel(
//			final N5Writer n5,
//			final String dataset,
//			final long[] blockPos,
//			final long addIfNonZero) throws IOException {
//		relabel(n5, dataset, blockPos, id -> id == 0 ? 0 : id + addIfNonZero);
//	}
//
//	private static void relabel(
//			final N5Writer n5,
//			final String dataset,
//			final long[] blockPos,
//			final LongUnaryOperator idMapping) throws IOException {
//		final DatasetAttributes attributes = n5.getDatasetAttributes(dataset);
//		final LongArrayDataBlock block = ((LongArrayDataBlock) n5.readBlock(dataset, attributes, blockPos));
//		final long[] data = block.getData();
//		for (int i = 0; i < data.length; ++i) {
//			data[i] = idMapping.applyAsLong(data[i]);
//		}
//		n5.writeBlock(dataset, attributes, new LongArrayDataBlock(block.getSize(), data, block.getGridPosition()));
//	}
//	private static void prepareOutputDatasets(
//			final N5Writer n5,
//			final Map<String, DatasetAttributes> datasets,
//			final Map<String, Object> additionalData
//	) throws IOException {
//
//		datasets.forEach(ThrowingBiConsumer.unchecked((ds, dt) -> prepareOutputDataset(n5, ds, dt, additionalData)));
//	}
//
//	private static void prepareOutputDataset(
//			final N5Writer n5,
//			final String dataset,
//			final DatasetAttributes attributes,
//			final Map<String, ?> additionalAttributes) throws IOException {
//		n5.createDataset(dataset, attributes);
//		for (Map.Entry<String, ?> entry : additionalAttributes.entrySet())
//			n5.setAttribute(dataset, entry.getKey(), entry.getValue());
//	}
//
//	private static <K, V> Map<K, V> with(Map<K, V> map, K key, V value) {
//		map.put(key, value);
//		return map;
//	}
//
//	private static class N5WriterSupplier implements Supplier<N5Writer>, Serializable {
//
//		private final String container;
//
//		private final boolean withPrettyPrinting;
//
//		private final boolean disableHtmlEscaping;
//
//		private final boolean serializeSpecialFloatingPointValues = true;
//
//		private N5WriterSupplier(final String container, final boolean withPrettyPrinting, final boolean disableHtmlEscaping) {
//			this.container = container;
//			this.withPrettyPrinting = withPrettyPrinting;
//			this.disableHtmlEscaping = disableHtmlEscaping;
//		}
//
//		@Override
//		public N5Writer get() {
//
//			try {
//				return Files.isDirectory(Paths.get(container))
//						? new N5FSWriter(container, createaBuilder())
//						: new N5HDF5Writer(container);
//			} catch (final IOException e) {
//				throw new RuntimeException(e);
//			}
//		}
//
//		private GsonBuilder createaBuilder() {
//			return serializeSpecialFloatingPointValues(withPrettyPrinting(disableHtmlEscaping(new GsonBuilder())));
//		}
//
//		private GsonBuilder serializeSpecialFloatingPointValues(final GsonBuilder builder) {
//			return with(builder, this.serializeSpecialFloatingPointValues, GsonBuilder::serializeSpecialFloatingPointValues);
//		}
//
//		private GsonBuilder withPrettyPrinting(final GsonBuilder builder) {
//			return with(builder, this.withPrettyPrinting, GsonBuilder::setPrettyPrinting);
//		}
//
//		private GsonBuilder disableHtmlEscaping(final GsonBuilder builder) {
//			return with(builder, this.disableHtmlEscaping, GsonBuilder::disableHtmlEscaping);
//		}
//
//		private static GsonBuilder with(final GsonBuilder builder, boolean applyAction, Function<GsonBuilder, GsonBuilder> action) {
//			return applyAction ? action.apply(builder) : builder;
//		}
//	}
//
//	private static double[] ones(final int length) {
//		double[] ones = new double[length];
//		Arrays.fill(ones, 1.0);
//		return ones;
//	}
//
//	private static Interval addDimension(final Interval interval, final long m, final long M) {
//		long[] min = new long[interval.numDimensions() + 1];
//		long[] max = new long[interval.numDimensions() + 1];
//		for (int d = 0; d < interval.numDimensions(); ++d) {
//			min[d] = interval.min(d);
//			max[d] = interval.max(d);
//		}
//		min[interval.numDimensions()] = m;
//		max[interval.numDimensions()] = M;
//		return new FinalInterval(min, max);
//	}
//
//	private static String toString(final Interval interval) {
//		return String.format("(%s %s)", Arrays.toString(Intervals.minAsLongArray(interval)), Arrays.toString(Intervals.maxAsLongArray(interval)));
//	}
//
//	private static double[] reverted(final double[] array, final boolean revert) {
//		return revert ? reverted(array) : array;
//	}
//
//	private static double[] reverted(final double[] array) {
//		final double[] copy = new double[array.length];
//		for (int i = 0, k = copy.length - 1; i < copy.length; ++i, --k) {
//			copy[i] = array[k];
//		}
//		return copy;
//	}
//
//	private static <T extends RealType<T>> ArrayImg<FloatType, FloatArray> smooth(
//			final RandomAccessibleInterval<T> source,
//			final Interval interval,
//			final int channelDim,
//			double sigma) {
//		final ArrayImg<FloatType, FloatArray> img = ArrayImgs.floats(Intervals.dimensionsAsLongArray(interval));
//
//		for (long channel = interval.min(channelDim); channel <= interval.max(channelDim); ++channel) {
//			Gauss3.gauss(
//					sigma,
//					Views.extendBorder(Views.hyperSlice(source, channelDim, channel)),
//					Views.hyperSlice(Views.translate(img, Intervals.minAsLongArray(interval)), channelDim, channel));
//		}
//		return img;
//	}
//
//	private static <T extends RealType<T>> void invalidateOutOfBlockAffinities(
//			final RandomAccessibleInterval<T> affs,
//			final T invalid,
//			final long[]... offsets
//	) {
//		for (int index = 0; index < offsets.length; ++index) {
//			final IntervalView<T> slice = Views.hyperSlice(affs, affs.numDimensions() - 1, index);
//			for (int d = 0; d < offsets[index].length; ++d) {
//				final long offset = offsets[index][d];
//				if (offset == 0)
//					continue;
//				final long pos = offset > 0 ? slice.max(d) + 1 - offset : slice.min(d) - 1 - offset;
//				Views.hyperSlice(slice, d, pos).forEach(p -> p.set(invalid));
//			}
//		}
//	}
//
//	private static <UF extends UnionFind> UF findOverlappingLabelsThresholdMedianEdgeAffinities(
//			final JavaSparkContext sc,
//			final Supplier<? extends N5Reader> labelContainer,
//			final Supplier<? extends N5Reader> affinitiesContainer,
//			final String labelDataset,
//			final String affinitiesDataset,
//			final double threshold,
//			final UF uf,
//			final long[] dimensions,
//			final int[] blockSize,
//			final int[] blocksPerTask,
//			final long... ignoreThese) {
//
//		Objects.requireNonNull(sc);
//		Objects.requireNonNull(blockSize);
//		Objects.requireNonNull(blocksPerTask);
//		Objects.requireNonNull(ignoreThese);
//
//		// TODO: 2/20/19
//
//		final int[] taskBlockSize = new int[blockSize.length];
//		Arrays.setAll(taskBlockSize, d -> blockSize[d] * blocksPerTask[d]);
//
//		final List<Tuple2<long[], long[]>> doubleSizeBlocks = Grids
//				.collectAllContainedIntervals(dimensions, taskBlockSize)
//				.stream()
//				.map(interval -> SparkMutexWatersheds.toMinMaxTuple(interval, Tuple2::new))
//				.collect(Collectors.toList());
//
//		final List<Tuple2<long[], long[]>> mappings = sc
//				.parallelize(doubleSizeBlocks)
//				.mapToPair(minMax -> {
//					final long[] thisBlockMax = minMax._2();
//
//					final RandomAccessibleInterval<UnsignedLongType> labels = N5Utils.open(labelContainer.get(), labelDataset);
//					final RandomAccessibleInterval<UnsignedLongType> thisBlockLabels = Views.interval(labels, minMax._1(), thisBlockMax);
//					final RandomAccessibleInterval<FloatType> affinities = N5Utils.open(affinitiesContainer.get(), affinitiesDataset);
//
//
//					final TLongSet ignoreTheseSet = new TLongHashSet(ignoreThese);
//
//					final TLongLongHashMap mapping = new TLongLongHashMap();
//					final UnionFind localUF = new LongHashMapUnionFind(mapping, 0, Long::compare);
//
//					for (int dim = 0; dim < thisBlockMax.length; ++dim) {
//						final long thisSliceIndex = thisBlockMax[dim];
//						final long thatSliceIndex = thisSliceIndex + 1;
//						if (thatSliceIndex >= dimensions[dim]) {
//							LOG.debug("That slice index {} outside dimensions {}", thatSliceIndex, dimensions);
//							continue;
//						}
//
//						LOG.debug("Slicing this block to {} and that block to {} for dim {}", thatSliceIndex, thatSliceIndex, dim);
//						// TODO once Intervals.hyperSlice is available on imglib2 release, we do not need to use Views.hyperSlice here
//						// imglib/imglib2@62998cd
//						final Interval interval = Views.hyperSlice(thisBlockLabels, dim, thisBlockLabels.min(dim));
//						RandomAccessibleInterval<UnsignedLongType> thisSliceLabels = Views.interval(Views.hyperSlice(labels, dim, thisSliceIndex), interval);
//						RandomAccessibleInterval<UnsignedLongType> thatSliceLabels = Views.interval(Views.hyperSlice(labels, dim, thatSliceIndex), interval);
//						RandomAccessibleInterval<FloatType> thisSliceAffinities = Views.interval(Views.hyperSlice(affinities, dim, thisSliceIndex), interval);
//						RandomAccessibleInterval<FloatType> thatSliceAffinities = Views.interval(Views.hyperSlice(affinities, dim, thatSliceIndex), interval);
//						LOG.debug("This slice: ({} {})", Intervals.minAsLongArray(thisSliceLabels), Intervals.maxAsLongArray(thisSliceLabels));
//						LOG.debug("That slice: ({} {})", Intervals.minAsLongArray(thatSliceLabels), Intervals.maxAsLongArray(thatSliceLabels));
//
//						final Cursor<UnsignedLongType> thisCursorLabels = Views.flatIterable(thisSliceLabels).cursor();
//						final Cursor<UnsignedLongType> thatCursorLabels = Views.flatIterable(thatSliceLabels).cursor();
//
//						final Cursor<FloatType> thisCursorAffinities = Views.flatIterable(thisSliceAffinities).cursor();
//						final Cursor<FloatType> thatCursorAffiniities = Views.flatIterable(thatSliceAffinities).cursor();
//
//						final TLongObjectMap<TLongObjectMap<TDoubleArrayList>> affinitiesByEdge = new TLongObjectHashMap<>();
//						final LongFunction<TDoubleArrayList> emptyListFactory = key -> new TDoubleArrayList();
//						final LongFunction<TLongObjectMap<TDoubleArrayList>> emptyMapFactory = key -> new TLongObjectHashMap<>();
//
//						while (thisCursorLabels.hasNext()) {
//							final long thisLabel = thisCursorLabels.next().getIntegerLong();
//							final long thatLabel = thatCursorLabels.next().getIntegerLong();
//
//							final double thisAffinity = thisCursorAffinities.next().getRealDouble();
//							final double thatAffinity = thatCursorAffiniities.next().getRealDouble();
//
//							if (ignoreTheseSet.contains(thisLabel) || ignoreTheseSet.contains(thatLabel))
//								continue;
//
//							if (thisLabel == thatLabel) {
//								LOG.error(
//										"Found same label {} in slices {} and {} for dimension {}",
//										thisLabel,
//										thisSliceIndex,
//										thatSliceIndex);
//								throw new RuntimeException("Got the same label in two different blocks -- impossible: " + thisLabel);
//							}
//
//							final long e1, e2;
//							if (thisLabel < thatLabel) {
//								e1 = thisLabel;
//								e2 = thatLabel;
//							}
//							else {
//								e1 = thatLabel;
//								e2 = thisLabel;
//							}
//
//							if (!affinitiesByEdge.containsKey(e1))
//								affinitiesByEdge.put(e1, new TLongObjectHashMap<>());
//
//							final TLongObjectMap<TDoubleArrayList> allNeighborAffinities = computeIfAbsent(affinitiesByEdge, e1, emptyMapFactory);
//							final TDoubleArrayList neighborAffinities = computeIfAbsent(allNeighborAffinities, e2, emptyListFactory);
//
//							if (!Double.isNaN(thisAffinity))
//								neighborAffinities.add(thisAffinity);
//
//							if (!Double.isNaN(thatAffinity))
//								neighborAffinities.add(thatAffinity);
//
//						}
//
//						LOG.info("Edge affinities: {}", affinitiesByEdge);
//
//
//						affinitiesByEdge.forEachEntry((k, v) -> {
//							TLongObjectIterator<TDoubleArrayList> edgeIt = v.iterator();
//							while (edgeIt.hasNext()) {
//								edgeIt.advance();
//								double[] affinitiesSorted = edgeIt.value().toArray();
//								if (affinitiesSorted.length > 0) {
//									Arrays.sort(affinitiesSorted);
//									if (affinitiesSorted[affinitiesSorted.length / 2] > threshold) {
//										localUF.join(localUF.findRoot(k), localUF.findRoot(edgeIt.key()));
//									}
//								}
//							}
//							return true;
//						});
//
//					}
//
//					LOG.info("Returning mapping {}", mapping);
//					return new Tuple2<>(minMax, new Tuple2<>(mapping.keys(), mapping.values()));
//				})
//				.values()
//				.collect();
//
//		for (final Tuple2<long[], long[]> mapping : mappings) {
//			final long[] keys = mapping._1();
//			final long[] vals = mapping._2();
//			for (int index = 0; index < keys.length; ++index) {
//				final long k = keys[index];
//				final long v = vals[index];
//				final long r1 = uf.findRoot(k);
//				final long r2 = uf.findRoot(v);
//				if (r1 != r2) {
//					uf.join(r1, r2);
//					uf.findRoot(r1);
//					uf.findRoot(r2);
//				}
//
//			}
//		}
//
//		return uf;
//
//
//	}
//
//	private static <UF extends UnionFind> UF findOverlappingLabelsArgMaxNoHalo(
//			final JavaSparkContext sc,
//			final Supplier<? extends N5Reader> n5,
//			final String labelDataset,
//			final UF uf,
//			final long[] dimensions,
//			final int[] blockSize,
//			final int[] blocksPerTask,
//			final long... ignoreThese) {
//
//		Objects.requireNonNull(sc);
//		Objects.requireNonNull(blockSize);
//		Objects.requireNonNull(blocksPerTask);
//		Objects.requireNonNull(ignoreThese);
//
//		final int[] taskBlockSize = new int[blockSize.length];
//		Arrays.setAll(taskBlockSize, d -> blockSize[d] * blocksPerTask[d]);
//
//		final List<Tuple2<long[], long[]>> doubleSizeBlocks = Grids
//				.collectAllContainedIntervals(dimensions, taskBlockSize)
//				.stream()
//				.map(interval -> SparkMutexWatersheds.toMinMaxTuple(interval, Tuple2::new))
//				.collect(Collectors.toList());
//
//		final List<Tuple2<long[], long[]>> mappings = sc
//				.parallelize(doubleSizeBlocks)
//				.mapToPair(minMax -> {
//					final long[] thisBlockMax = minMax._2();
//
//					final RandomAccessibleInterval<UnsignedLongType> labels = N5Utils.open(n5.get(), labelDataset);
//					final RandomAccessibleInterval<UnsignedLongType> thisBlock = Views.interval(labels, minMax._1(), thisBlockMax);
//
//					final TLongSet ignoreTheseSet = new TLongHashSet(ignoreThese);
//
//					final TLongLongHashMap mapping = new TLongLongHashMap();
//					final UnionFind localUF = new LongHashMapUnionFind(mapping, 0, Long::compare);
//
//					for (int dim = 0; dim < thisBlockMax.length; ++dim) {
//						final long thisSliceIndex = thisBlockMax[dim];
//						final long thatSliceIndex = thisSliceIndex + 1;
//						if (thatSliceIndex >= dimensions[dim]) {
//							LOG.debug("That slice index {} outside dimensions {}", thatSliceIndex, dimensions);
//							continue;
//						}
//
//						LOG.debug("Slicing this block to {} and that block to {} for dim {}", thatSliceIndex, thatSliceIndex, dim);
//						final Interval interval = Views.hyperSlice(thisBlock, dim, thisBlock.min(dim));
//						RandomAccessibleInterval<UnsignedLongType> thisSlice = Views.interval(Views.hyperSlice(labels, dim, thisSliceIndex), interval);
//						RandomAccessibleInterval<UnsignedLongType> thatSlice = Views.interval(Views.hyperSlice(labels, dim, thatSliceIndex), interval);
//						LOG.debug("This slice: ({} {})", Intervals.minAsLongArray(thisSlice), Intervals.maxAsLongArray(thisSlice));
//						LOG.debug("That slice: ({} {})", Intervals.minAsLongArray(thatSlice), Intervals.maxAsLongArray(thatSlice));
//
//						final Cursor<UnsignedLongType> thisCursor = Views.flatIterable(thisSlice).cursor();
//						final Cursor<UnsignedLongType> thatCursor = Views.flatIterable(thatSlice).cursor();
//
//						final TLongObjectMap<TLongIntMap> thisMap = new TLongObjectHashMap<>();
//						final TLongObjectMap<TLongIntMap> thatMap = new TLongObjectHashMap<>();
//
//						while (thisCursor.hasNext()) {
//							final long thisLabel = thisCursor.next().getIntegerLong();
//							final long thatLabel = thatCursor.next().getIntegerLong();
//
//							if (ignoreTheseSet.contains(thisLabel) || ignoreTheseSet.contains(thatLabel))
//								continue;
//
//							if (thisLabel == thatLabel) {
//								LOG.error(
//										"Found same label {} in slices {} and {} for dimension {}",
//										thisLabel,
//										thisSliceIndex,
//										thatSliceIndex);
//								throw new RuntimeException("Got the same label in two different blocks -- impossible: " + thisLabel);
//							}
//
//							if (!thisMap.containsKey(thisLabel))
//								thisMap.put(thisLabel, new TLongIntHashMap());
//
//							if (!thatMap.containsKey(thatLabel))
//								thatMap.put(thatLabel, new TLongIntHashMap());
//
//							addOne(thisMap.get(thisLabel), thatLabel);
//							addOne(thatMap.get(thatLabel), thisLabel);
//
////							thatArgMax.forEachEntry((k, v) -> {
//////								if (thatArgMax.get(v) == k)
////								localUF.join(localUF.findRoot(v), localUF.findRoot(k));
//////								mapping.put(k, v);
////								return true;
////							});
//
//
//						}
//
//						LOG.debug("Mapping this to that {}", thisMap);
//						LOG.debug("Mapping that to this {}", thatMap);
//
//						final TLongLongMap thisArgMax = argMaxCounts(thisMap);
//						final TLongLongMap thatArgMax = argMaxCounts(thatMap);
//
//						thisArgMax.forEachEntry((k, v) -> {
//							if (thatArgMax.get(v) == k)
//								localUF.join(localUF.findRoot(v), localUF.findRoot(k));
//							return true;
//						});
//
//					}
//
//					LOG.debug("Returning mapping {}", mapping);
//					return new Tuple2<>(minMax, new Tuple2<>(mapping.keys(), mapping.values()));
//				})
//				.values()
//				.collect();
//
//		for (final Tuple2<long[], long[]> mapping : mappings) {
//			final long[] keys = mapping._1();
//			final long[] vals = mapping._2();
//			for (int index = 0; index < keys.length; ++index) {
//				final long k = keys[index];
//				final long v = vals[index];
//				final long r1 = uf.findRoot(k);
//				final long r2 = uf.findRoot(v);
//				if (r1 != r2) {
//					uf.join(r1, r2);
//					uf.findRoot(r1);
//					uf.findRoot(r2);
//				}
//
//			}
//		}
//
//		return uf;
//
//
//	}
//
//	private static class CropAffinities implements PairFunction<Interval, Interval, RandomAccessibleInterval<FloatType>> {
//
//		private final Supplier<? extends N5Reader> n5in;
//
//		private final Supplier<? extends N5Reader> maskN5;
//
//		private final Supplier<? extends N5Reader> gliaN5;
//
//		private final String affinities;
//
//		private final String maskDataset;
//
//		private final String gliaDataset;
//
//		private final long[][] offsets;
//
//		private final long[] halo;
//
//		private CropAffinities(
//				final Supplier<? extends N5Reader> n5in,
//				final Supplier<? extends N5Reader> maskN5,
//				final Supplier<? extends N5Reader> gliaN5,
//				final String affinities,
//				final String maskDataset,
//				final String gliaDataset,
//				final long[][] offsets,
//				final long[] halo) {
//			this.n5in = n5in;
//			this.maskN5 = maskN5;
//			this.gliaN5 = gliaN5;
//			this.affinities = affinities;
//			this.maskDataset = maskDataset;
//			this.gliaDataset = gliaDataset;
//			this.halo = halo;
//			this.offsets = offsets;
//		}
//
//		@Override
//		public Tuple2<Interval, RandomAccessibleInterval<FloatType>> call(final Interval interval) throws Exception {
//
//			final RandomAccessibleInterval<FloatType> affs = N5Utils.open(n5in.get(), affinities);
//
//			final RandomAccessible<FloatType> glia = gliaN5 == null || gliaN5.get() == null || gliaDataset == null
//					? ConstantUtils.constantRandomAccessible(new FloatType(0.0f), affs.numDimensions() - 1)
//					: Views.extendZero(N5Utils.<FloatType>open(gliaN5.get(), gliaDataset));
//
//			final RandomAccessible<UnsignedLongType> mask = maskN5 == null || maskN5.get() == null || gliaDataset == null
//					? ConstantUtils.constantRandomAccessible(new UnsignedLongType(1), affs.numDimensions() - 1)
//					: Views.extendZero(N5Utils.<UnsignedLongType>open(maskN5.get(), maskDataset));
//
//
//			final Interval withHalo = Intervals.expand(interval, halo);
//
//			final long[] min = new long[withHalo.numDimensions() + 1];
//			final long[] max = new long[withHalo.numDimensions() + 1];
//			withHalo.min(min);
//			withHalo.max(max);
//			min[min.length - 1] = affs.min(min.length - 1);
//			max[max.length - 1] = affs.max(max.length - 1);
//			final RandomAccessibleInterval<UnsignedLongType> fromMask = Views.interval(mask, withHalo);
//			final RandomAccessibleInterval<FloatType> gliaWithHalo = Views.interval(glia, withHalo);
//			final Interval targetInterval = new FinalInterval(min, max);
//			final ArrayImg<FloatType, FloatArray> affinityCrop = ArrayImgs.floats(Intervals.dimensionsAsLongArray(targetInterval));
//
//			for (int i = 0; i < offsets.length; ++i) {
//				final long[] offset = offsets[i];
//				final RandomAccessibleInterval<FloatType> affinityCropSlice = Views.hyperSlice(affinityCrop, affs.numDimensions() - 1, (long) i);
//				final RandomAccessibleInterval<FloatType> affinitySlice = Views.interval(Views.extendValue(Views.hyperSlice(affs, affs.numDimensions() - 1, (long) i), new FloatType(Float.NaN)), withHalo);
//				final RandomAccessibleInterval<UnsignedLongType> toMask = Views.interval(Views.extendZero(fromMask), Intervals.translate(withHalo, offset));
//				final RandomAccessibleInterval<FloatType> fromGlia = Views.interval(glia, withHalo);
//				final RandomAccessibleInterval<FloatType> toGlia = Views.interval(glia, Intervals.translate(withHalo, offset));
//
//				LoopBuilder
//						.setImages(affinitySlice, affinityCropSlice, fromMask, toMask, fromGlia, toGlia)
//						.forEachPixel((s, t, fM, tM, fG, tG) -> {
//							final boolean isOutsideMask = fM.getIntegerLong() == 0 || tM.getIntegerLong() == 0;
//							if (isOutsideMask)
//								t.setReal(Double.NaN);
//							else {
//								final double gliaWeight = Math.max(Math.min(1.0 - fG.getRealDouble(), 1.0), 0.0) * (1.0 - Math.max(Math.min(1.0 - tG.getRealDouble(), 1.0), 0.0));
//								t.setReal(gliaWeight * s.getRealDouble());
//							}
//						});
//
//			}
//
//
//			return new Tuple2<>(interval, affinityCrop);
//		}
//	}
//
//	private static <T> T toMinMaxTuple(final Interval interval, BiFunction<long[], long[], T> toTuple) {
//		return toTuple.apply(Intervals.minAsLongArray(interval), Intervals.maxAsLongArray(interval));
//	}
//
//	private static void addOne(final TLongIntMap countMap, final long label) {
//		countMap.put(label, countMap.get(label) + 1);
//	}
//
//	private static TLongLongMap argMaxCounts(final TLongObjectMap<TLongIntMap> counts) {
//		final TLongLongMap mapping = new TLongLongHashMap();
//		counts.forEachEntry((k, v) -> {
//			mapping.put(k, argMaxCount(v));
//			return true;
//		});
//		return mapping;
//	}
//
//	private static long argMaxCount(final TLongIntMap counts) {
//		long maxCount = Long.MIN_VALUE;
//		long argMaxCount = 0;
//		for (final TLongIntIterator it = counts.iterator(); it.hasNext(); ) {
//			it.advance();
//			final long v = it.value();
//			if (v > maxCount) {
//				maxCount = v;
//				argMaxCount = it.key();
//			}
//		};
//		return argMaxCount;
//	}
//
//	private static class CachedMapper implements LongUnaryOperator {
//
//
//		private long nextId;
//
//		private final TLongLongMap cache = new TLongLongHashMap();
//
//		private CachedMapper(final long firstId) {
//			this.nextId = firstId;
//		}
//
//		@Override
//		public long applyAsLong(long l) {
//
//			if (l == 0)
//				return 0;
//
//			if (!cache.containsKey(l))
//				cache.put(l, nextId++);
//
//			return cache.get(l);
//		}
//	}
//
//	private static <T> T computeIfAbsent(final TLongObjectMap<T> map, final long key, final LongFunction<T> mappingFactory) {
//		final T value = map.get(key);
//		if (value != null)
//			return value;
//		final T t = mappingFactory.apply(key);
//		map.put(key, t);
//		return t;
//	}
//
//	private static boolean isNearestNeighbor(long[] offset) {
//		return Arrays.stream(offset).map(o -> o * o).sum() <= 1;
//	}

}
