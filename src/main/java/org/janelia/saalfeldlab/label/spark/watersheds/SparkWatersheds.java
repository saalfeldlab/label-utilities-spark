package org.janelia.saalfeldlab.label.spark.watersheds;

import com.google.gson.GsonBuilder;
import com.google.gson.annotations.Expose;
import com.pivovarit.function.ThrowingBiConsumer;
import gnu.trove.iterator.TLongIntIterator;
import gnu.trove.iterator.TLongObjectIterator;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TLongIntMap;
import gnu.trove.map.TLongLongMap;
import gnu.trove.map.TLongObjectMap;
import gnu.trove.map.hash.TLongIntHashMap;
import gnu.trove.map.hash.TLongLongHashMap;
import gnu.trove.map.hash.TLongObjectHashMap;
import gnu.trove.set.TLongSet;
import gnu.trove.set.hash.TLongHashSet;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.algorithm.labeling.Watersheds;
import net.imglib2.algorithm.localextrema.LocalExtrema;
import net.imglib2.algorithm.neighborhood.DiamondShape;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.algorithm.util.Singleton;
import net.imglib2.algorithm.util.unionfind.LongHashMapUnionFind;
import net.imglib2.algorithm.util.unionfind.UnionFind;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.array.ArrayRandomAccess;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.message.BasicNameValuePair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.janelia.saalfeldlab.label.spark.N5Helpers;
import org.janelia.saalfeldlab.label.spark.Version;
import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.LongFunction;
import java.util.function.LongUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SparkWatersheds {

	private static final String RESOLUTION_KEY = "resolution";

	private static final String OFFSET_KEY = "offset";

	private static final String ARGUMENTS_KEY = "arguments";

	private static final String ARGV_KEY = "argumentVector";

	private static final String LABEL_UTILITIES_SPARK_KEY = "label-utilities-spark";

	private static final String VERSION_KEY = "version";

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	private static class Args implements Serializable, Callable<Void> {

		@Expose
		@CommandLine.Parameters(arity = "1", paramLabel = "INPUT_CONTAINER", description = "Path to N5 container with averagedAffinities dataset.")
		String inputContainer = null;

		@Expose
		@CommandLine.Option(names = "--output-container", paramLabel = "OUTPUT_CONTAINER", description = "Path to output container. Defaults to INPUT_CONTAINER.")
		String outputContainer = null;

		@Expose
		@CommandLine.Option(names = "--averaged-affinity-dataset", paramLabel = "AFFINITIES", description = "Path of averagedAffinities dataset in INPUT_CONTAINER.")
		String averagedAffinities = "volumes/averagedAffinities/prediction-average";

		@Expose
		@CommandLine.Option(names = "--label-datasets-prefix", paramLabel = "LABEL_DATASETS_PREFIX", defaultValue = "volumes/labels", description = "Will be prepended to datasets: ${LABEL_DATASETS_PREFIX}/${LABEL_DATASET}. Set to empty String to ignore.")
		String labelDatasetsPrefix;

		@Expose
		@CommandLine.Option(names = "--watershed-seeds-dataset", description = "Path in container: ${LABEL_DATASETS_PREFIX}/${WATERSHED_SEEDS_DATASET", defaultValue = "watershed-seeds")
		String watershedSeeds = "watershed-seeds";

		@Expose
		@CommandLine.Option(names = "--seeded-watersheds-dataset", paramLabel = "WATERSHEDS", description = "Path in container: ${LABEL_DATASETS_PREFIX}/${WATERSHEDS}", defaultValue = "seeded-watersheds")
		String seededWatersheds;

		@Expose
		@CommandLine.Option(names = "--merged-dataset", paramLabel = "WATERSHEDS_MERGED", description = "Path in container: ${LABEL_DATASETS_PREFIX}/${WATERSHEDS_MERGED}", defaultValue = "watersheds-merged")
		String merged;

		@Expose
		@CommandLine.Option(names = "--block-merged-dataset", paramLabel = "MERGED", description = "Path in container: ${LABEL_DATASETS_PREFIX}/${MERGED}", defaultValue = "merged")
		String blockMerged;

		@Expose
		@CommandLine.Option(names = "--block-size", paramLabel = "BLOCK_SIZE", description = "Block size of output.", split = ",")
		int[] blockSize = {64, 64, 64};

		@Expose
		@CommandLine.Option(names = "--blocks-per-task", paramLabel = "BLOCKS_PER_TASK", description = "How many blocks to combine for watersheds/connected components (one value per dimension)", split = ",")
		int[] blocksPerTask = {1, 1, 1};

		@Expose
		@CommandLine.Option(names = "--threshold", paramLabel = "THRESHOLD", description = "Threshold for thresholding affinities.", defaultValue = "0.9")
		Double threshold;

		@Expose
		@CommandLine.Option(names = "--minimum-watershed-affinity", paramLabel = "THRESHOLD", description = "Threshold for thresholding affinities.")
		Double minimumAffinity = Double.NEGATIVE_INFINITY;

		@Expose
		@CommandLine.Option(names = "--halo", paramLabel = "HALO", description = "Include halo region to run connected components/watersheds", split = ",")
		int[] halo = {0, 0, 0};

		@Expose
		@CommandLine.Option(names = "--cropped-datasets-pattern", paramLabel = "CROPPED_DATASETS_PATTERN", description = "All data sets are stored with halo by default. Cropped versions according to this.", defaultValue = "%s-cropped")
		String croppedDatasetPattern;

		@Expose
		@CommandLine.Option(names = "--relabel", paramLabel = "RELABEL", description = "Relabel all label data sets for unique labels", defaultValue = "false")
		Boolean relabel;

		@Expose
		@CommandLine.Option(names = "--reverse-array-attributes", paramLabel = "RELABEL", description = "Reverse all array attributes (that are not dataset attributes)", defaultValue = "false")
		Boolean reverseArrayAttributes;

		@CommandLine.Option(names = "--json-pretty-print", defaultValue = "true")
		transient Boolean prettyPrint;

		@CommandLine.Option(names = "--json-disable-html-escape", defaultValue = "true")
		transient Boolean disbaleHtmlEscape;

		@CommandLine.Option(names = {"-h", "--help"}, usageHelp = true, description = "Display this help and exit")
		private Boolean help;

		@Override
		public Void call() {

			watershedSeeds = String.join("/", labelDatasetsPrefix, watershedSeeds);
			seededWatersheds = String.join("/", labelDatasetsPrefix, seededWatersheds);
			merged = String.join("/", labelDatasetsPrefix, merged);
			blockMerged = String.join("/", labelDatasetsPrefix, blockMerged);
			return null;
		}
	}

	public static void main(final String[] argv) throws IOException {

		run(argv);

	}

	public static void run(final String... argv) throws IOException {

		final Args args = new Args();
		CommandLine.call(args, argv);

		final N5WriterSupplier n5in = new N5WriterSupplier(args.inputContainer, args.prettyPrint, args.disbaleHtmlEscape);

		final N5WriterSupplier n5out = args.outputContainer == null
				? n5in
				: new N5WriterSupplier(args.outputContainer, args.prettyPrint, args.disbaleHtmlEscape);

		final DatasetAttributes inputAttributes = n5in.get().getDatasetAttributes(args.averagedAffinities);
		final long[] inputDims = inputAttributes.getDimensions();
		final long[] outputDims = inputDims.clone();

		final Map<String, Object> labelUtilitiesSparkAttributes = new HashMap<>();
		labelUtilitiesSparkAttributes.put(ARGUMENTS_KEY, args);
		labelUtilitiesSparkAttributes.put(ARGV_KEY, argv);
		labelUtilitiesSparkAttributes.put(VERSION_KEY, Version.VERSION_STRING);
		final Map<String, Object> attributes = with(new HashMap<>(), LABEL_UTILITIES_SPARK_KEY, labelUtilitiesSparkAttributes);

		final int[] taskBlockSize = IntStream.range(0, args.blockSize.length).map(d -> args.blockSize[d] * args.blocksPerTask[d]).toArray();
		final boolean hasHalo = Arrays.stream(args.halo).filter(h -> h != 0).count() > 0;
		if (hasHalo)
			throw new UnsupportedOperationException("Halo currently not supported, please omit halo option!");

		String[] uint64Datasets = {args.merged, args.seededWatersheds, args.watershedSeeds, args.blockMerged};
		String[] uint8Datasets = {};

		final double[] resolution = reversed(Optional.ofNullable(n5in.get().getAttribute(args.averagedAffinities, RESOLUTION_KEY, double[].class)).orElse(ones(outputDims.length)), args.reverseArrayAttributes);
		final double[] offset = reversed(Optional.ofNullable(n5in.get().getAttribute(args.averagedAffinities, OFFSET_KEY, double[].class)).orElse(new double[outputDims.length]), args.reverseArrayAttributes);
		attributes.put(RESOLUTION_KEY, resolution);
		attributes.put(OFFSET_KEY, offset);

		final Map<String, DatasetAttributes> datasets = new HashMap<>();
		Arrays.asList(uint64Datasets).forEach(ds -> datasets.put(ds, new DatasetAttributes(outputDims, args.blockSize, DataType.UINT64, new GzipCompression())));
		Arrays.asList(uint8Datasets).forEach(ds -> datasets.put(ds, new DatasetAttributes(outputDims, args.blockSize, DataType.UINT8, new GzipCompression())));

		if (hasHalo) {
			prepareOutputDatasets(
					n5out.get(),
					datasets.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, e -> new DatasetAttributes(e.getValue().getDimensions(), taskBlockSize, e.getValue().getDataType(), e.getValue().getCompression()))),
					attributes);
		}

		prepareOutputDatasets(
				n5out.get(),
				datasets.entrySet().stream().collect(Collectors.toMap(entry -> hasHalo ? String.format(args.croppedDatasetPattern, entry.getKey()) : entry.getKey(), Map.Entry::getValue)),
				attributes);

		final SparkConf conf = new SparkConf().setAppName(MethodHandles.lookup().lookupClass().getName());
		try (final JavaSparkContext sc = new JavaSparkContext(conf)) {

			LOG.info("Input  dims:   {}", inputDims);
			LOG.info("Output dims:   {}", outputDims);
			run(
					sc,
					n5in,
					n5out,
					outputDims,
					args.minimumAffinity,
					IntStream.of(args.halo).mapToLong(i -> i).toArray(),
					//					new SerializableMergeWatershedsMinThresholdSupplier(args.threshold),
					new SerializableMergeWatershedsMedianThresholdSupplier(args.threshold),
					args.threshold,
					args.averagedAffinities,
					args.merged,
					args.blockMerged,
					args.watershedSeeds,
					args.seededWatersheds,
					args.croppedDatasetPattern,
					args.blockSize,
					args.blocksPerTask,
					args.relabel);
		}

	}

	public static void run(
			final JavaSparkContext sc,
			final Supplier<? extends N5Reader> n5in,
			final Supplier<? extends N5Writer> n5out,
			final long[] outputDims,
			final double minimumWatershedAffinity,
			final long[] halo,
			final Supplier<MergeWatersheds> mergeWatershedregions,
			final double mergeFragmentsThreshold,
			final String averagedAffinities,
			final String merged,
			final String blockMerged,
			final String watershedSeeds,
			final String seededWatersheds,
			final String croppedDatasetPattern,
			final int[] blockSize,
			final int[] blocksPerTask,
			final boolean relabel) throws IOException {

		final boolean hasHalo = Arrays.stream(halo).filter(h -> h != 0).count() > 0;
		final int[] watershedBlockSize = IntStream.range(0, blockSize.length).map(d -> blockSize[d] * blocksPerTask[d]).toArray();
		final List<Tuple2<long[], long[]>> watershedBlocks = Grids
				.collectAllContainedIntervals(outputDims, watershedBlockSize)
				.stream()
				.map(i -> new Tuple2<>(Intervals.minAsLongArray(i), Intervals.maxAsLongArray(i)))
				.collect(Collectors.toList());
		;
		final long[] negativeHalo = new long[halo.length];
		Arrays.setAll(negativeHalo, d -> -halo[d]);

		final List<Tuple2<Tuple2<long[], long[]>, Integer>> idCounts = sc
				.parallelize(watershedBlocks)
				.map(t -> (Interval)new FinalInterval(t._1(), t._2()))
				.mapToPair(new CropAffinities(n5in, averagedAffinities, minimumWatershedAffinity, halo))
				.mapToPair(t -> {
					final Interval block = t._1();
					final RandomAccessibleInterval<FloatType> relief = t._2();
					// we expect that there are no NaNs in the relief. Replace NaNs with Double.NEGATIVE_INFINITY as appropriate
					// Any point that is surrounded entirely by NaN will be added as seed point by default. This is not the behavior
					// we would like to have
					List<Point> seeds = LocalExtrema
							.findLocalExtrema(
									Views.extendValue(relief, new FloatType(Float.MIN_VALUE)),
									relief,
									new LocalExtrema.MaximumCheck<>(new FloatType((float)minimumWatershedAffinity)));
					LOG.debug("Got {} seeds", seeds.size());

					final CellGrid grid = new CellGrid(outputDims, blockSize);
					final CellGrid watershedsGrid = new CellGrid(outputDims, watershedBlockSize);
					LOG.debug("Got grids {} and {}", grid, watershedsGrid);

					final long[] blockOffset = Intervals.minAsLongArray(block);
					final long[] watershedsBlockOffset = blockOffset.clone();
					grid.getCellPosition(blockOffset, blockOffset);
					watershedsGrid.getCellPosition(watershedsBlockOffset, watershedsBlockOffset);
					LOG.debug("min={} blockOffset={} watershedsBlockOffset={}", Intervals.minAsLongArray(block), blockOffset, watershedsBlockOffset);

					final long[] dims = Intervals.dimensionsAsLongArray(relief);
					final ArrayImg<UnsignedLongType, LongArray> labels = ArrayImgs.unsignedLongs(dims);
					{
						final ArrayRandomAccess<UnsignedLongType> labelsAccess = labels.randomAccess();
						for (int i = 0; i < seeds.size(); ++i) {
							labelsAccess.setPosition(seeds.get(i));
							labelsAccess.get().setInteger(i + 1L);
						}
					}

					final Interval relevantInterval = Intervals.expand(labels, negativeHalo);

					final DatasetAttributes croppedAttributes = new DatasetAttributes(outputDims, blockSize, DataType.UINT64, new GzipCompression());
					final DatasetAttributes watershedAttributes = new DatasetAttributes(outputDims, watershedBlockSize, DataType.UINT64, new GzipCompression());

					// TODO do seeded watersheds
					LOG.debug("Found watershed seeds {}", seeds);
					N5Utils.saveBlock(Views.interval(labels, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, croppedAttributes, blockOffset);
					if (hasHalo) {
						throw new UnsupportedOperationException("Need to implement halo support!");
						//						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(watershedSeedsMaskImg), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
						//						n5out.get().writeBlock(watershedSeeds, watershedAttributes, dataBlock);
					}

					N5Utils.saveBlock(Views.interval(labels, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, croppedAttributes, blockOffset);
					if (hasHalo) {
						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(labels), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
						n5out.get().writeBlock(watershedSeeds, watershedAttributes, dataBlock);
					}

					Watersheds.seededRealType(
							Views.extendValue(relief, new FloatType(Float.NaN)),
							labels,
							seeds,
							(value, ref) -> -value.getRealDouble(),
							new DiamondShape(1)
					);

					N5Utils.saveBlock(Views.interval(labels, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, croppedAttributes, blockOffset);
					if (hasHalo) {
						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(labels), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
						n5out.get().writeBlock(seededWatersheds, watershedAttributes, dataBlock);
					}

					final LongUnaryOperator mapping = mergeWatershedregions.get().getMapping(relief, labels, seeds.size() + 1);

					// TODO find better logic instead of label == 0 check
					final TLongSet ids = new TLongHashSet();
					final TLongIntHashMap counts = new TLongIntHashMap();
					for (final UnsignedLongType pix : labels) {
						final long label = pix.getIntegerLong();
						if (label == 0)
							counts.put(label, counts.get(label) + 1);
						else {
							final long root = mapping.applyAsLong(pix.getIntegerLong());
							ids.add(root);
							counts.put(root, counts.get(root) + 1);
							if (root != 0)
								pix.setInteger(root);
						}
					}

					for (final UnsignedLongType pix : labels) {
						if (counts.get(pix.getIntegerLong()) == 1) {
							counts.remove(pix.getIntegerLong());
							counts.put(0, counts.get(0) + 1);
							pix.setZero();
						}
					}

					N5Utils.saveBlock(Views.interval(labels, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, merged) : merged, croppedAttributes, blockOffset);
					if (hasHalo) {
						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(labels), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
						n5out.get().writeBlock(merged, watershedAttributes, dataBlock);
					}

					return new Tuple2<>(new Tuple2<>(Intervals.minAsLongArray(t._1()), Intervals.maxAsLongArray(t._1())), ids.size());
				})
				.collect();

		long startIndex = 1;
		final List<Tuple2<Tuple2<long[], long[]>, Long>> idOffsets = new ArrayList<>();
		for (final Tuple2<Tuple2<long[], long[]>, Integer> idCount : idCounts) {
			idOffsets.add(new Tuple2<>(idCount._1(), startIndex));
			startIndex += idCount._2();
		}

		if (relabel) {
			final long numBlocks = sc
					.parallelizePairs(idOffsets)
					.map(t -> {
						LOG.debug("Relabeling block ({} {}) starting at id {}", t._1()._1(), t._1()._2(), t._2());
						final N5Writer n5 = n5out.get();
						final Interval interval = new FinalInterval(t._1()._1(), t._1()._2());
						//						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, interval, t._2());
						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, merged) : merged, merged, interval, t._2());
						//						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, interval, t._2());
						if (hasHalo)
							throw new UnsupportedOperationException("Halo relabeling not implemented yet!");

						// TODO do halo relabeling

						return true;
					})
					.count();
			LOG.debug("Relabeled {} blocks", numBlocks);
			final long maxId = startIndex;
			LOG.info("Found max id {}", maxId);
			final N5Writer writer = n5out.get();
			writer.setAttribute(hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, "maxId", maxId);
			writer.setAttribute(hasHalo ? String.format(croppedDatasetPattern, merged) : merged, "maxId", maxId);
			writer.setAttribute(hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, "maxId", maxId);

			if (hasHalo)
				throw new UnsupportedOperationException("Halo relabeling not implemented yet!");

			if (maxId + 2 > Integer.MAX_VALUE)
				throw new RuntimeException("Currently only Integer.MAX_VALUE labels supported");

			//			final IntArrayUnionFind uf = findOverlappingLabelsArgMaxNoHalo(sc, n5out, merged, new IntArrayUnionFind((int) (maxId + 2)), outputDims, blockSize, blocksPerTask, 0);
			final LongHashMapUnionFind uf = findOverlappingLabelsThresholdMedianEdgeAffinities(
					sc,
					n5out,
					n5in,
					merged,
					averagedAffinities,
					mergeFragmentsThreshold,
					new LongHashMapUnionFind(),
					outputDims,
					blockSize,
					blocksPerTask,
					0);
			LOG.debug("{} sets are grouped into {} sets", uf.size(), uf.setCount());

			final List<Tuple2<Tuple2<long[], long[]>, Tuple2<long[], long[]>>> finalMappings = new ArrayList<>();
			long maxRoot = 0;
			for (int index = 0; index < idOffsets.size(); ++index) {
				final Tuple2<Tuple2<long[], long[]>, Long> idOffset = idOffsets.get(index);
				final long minLabel = idOffset._2();
				final long maxLabel = index < idOffsets.size() - 1
						? idOffsets.get(index + 1)._2()
						: (maxId);

				LOG.debug("Max label = {} min label = {} for block ({} {})", maxLabel, minLabel, idOffset._1()._1(), idOffset._1()._2());
				final long[] keys = new long[(int)(maxLabel - minLabel)];
				final long[] vals = new long[keys.length];

				for (int i = 0; i < keys.length; ++i) {
					final long k = i + minLabel;
					final long root = uf.findRoot(k);
					keys[i] = k;
					vals[i] = root;
					if (root > maxRoot) {
						maxRoot = root;
					}
				}
				finalMappings.add(new Tuple2<>(idOffset._1(), new Tuple2<>(keys, vals)));
			}
			n5out.get().setAttribute(blockMerged, "maxId", maxRoot);

			sc
					.parallelize(finalMappings)
					.foreach(t -> {
						final Interval interval = new FinalInterval(t._1()._1(), t._1()._2());
						final TLongLongMap mapping = new TLongLongHashMap(t._2()._1(), t._2()._2());

						//						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, interval, mapping);
						relabel(n5out.get(), hasHalo ? String.format(croppedDatasetPattern, merged) : merged, blockMerged, interval, mapping);
						//						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, interval, mapping);
						if (hasHalo)
							throw new UnsupportedOperationException("Halo relabeling not implemented yet!");
					});

		}

	}

	private static void relabel(
			final N5Writer n5,
			final String source,
			final String target,
			final Interval interval,
			final TLongLongMap mapping) throws IOException {

		SparkWatersheds.<UnsignedLongType>relabel(n5, source, target, interval, (src, tgt) -> {
			final long val = mapping.get(src.getIntegerLong());
			if (val != 0)
				tgt.set(val);
		});
	}

	private static void relabel(
			final N5Writer n5,
			final String source,
			final String target,
			final Interval interval,
			final long addIfNotZero) throws IOException {

		final CachedMapper mapper = new CachedMapper(addIfNotZero);
		SparkWatersheds.<UnsignedLongType>relabel(n5, source, target, interval, (src, tgt) -> tgt.set(mapper.applyAsLong(src.getIntegerLong())));
	}

	private static <T extends IntegerType<T> & NativeType<T>> void relabel(
			final N5Writer n5,
			final String source,
			final String target,
			final Interval interval,
			final BiConsumer<T, T> idMapping) throws IOException {

		final String writerCacheKey = new URIBuilder(n5.getURI()).setParameters(
				new BasicNameValuePair("type", "writer"),
				new BasicNameValuePair("call", "spark-watersheds-relabel")
		).toString();
		final var writer = Singleton.get(writerCacheKey, () -> n5);

		final DatasetAttributes attributes = writer.getDatasetAttributes(source);
		final CellGrid grid = new CellGrid(attributes.getDimensions(), attributes.getBlockSize());

		final String labelsCacheKey = new URIBuilder(writer.getURI()).setParameters(
				new BasicNameValuePair("call", "spark-watersheds-relabel"),
				new BasicNameValuePair("dataset", source)
		).toString();
		final var img = Singleton.get(labelsCacheKey, () -> N5Utils.<T>open(writer, source));

		final RandomAccessibleInterval<T> data = Views.interval(img, interval);
		final RandomAccessibleInterval<T> copy = new ArrayImgFactory<>(Util.getTypeFromInterval(data).createVariable()).create(data);
		LOG.debug("Input size {} -- output size {}", Intervals.dimensionsAsLongArray(data), Intervals.dimensionsAsLongArray(copy));
		long maxId = Long.MIN_VALUE;
		{
			final Cursor<T> src = Views.flatIterable(data).cursor();
			final Cursor<T> tgt = Views.flatIterable(copy).cursor();
			while (src.hasNext()) {
				idMapping.accept(src.next(), tgt.next());
				maxId = Math.max(tgt.get().getIntegerLong(), maxId);
			}
			LOG.debug("Remapped to max id {}", maxId);
		}
		final long[] blockPos = Intervals.minAsLongArray(interval);
		grid.getCellPosition(blockPos, blockPos);
		N5Utils.saveBlock(copy, writer, target, attributes, blockPos);
	}

	private static void relabel(
			final N5Writer n5,
			final String dataset,
			final long[] blockPos,
			final long addIfNonZero) throws IOException {

		relabel(n5, dataset, blockPos, id -> id == 0 ? 0 : id + addIfNonZero);
	}

	private static void relabel(
			final N5Writer n5,
			final String dataset,
			final long[] blockPos,
			final LongUnaryOperator idMapping) throws IOException {

		final DatasetAttributes attributes = n5.getDatasetAttributes(dataset);
		final LongArrayDataBlock block = ((LongArrayDataBlock)n5.readBlock(dataset, attributes, blockPos));
		final long[] data = block.getData();
		for (int i = 0; i < data.length; ++i) {
			data[i] = idMapping.applyAsLong(data[i]);
		}
		n5.writeBlock(dataset, attributes, new LongArrayDataBlock(block.getSize(), data, block.getGridPosition()));
	}

	private static void prepareOutputDatasets(
			final N5Writer n5,
			final Map<String, DatasetAttributes> datasets,
			final Map<String, Object> additionalData
	) throws IOException {

		datasets.forEach(ThrowingBiConsumer.unchecked((ds, dt) -> prepareOutputDataset(n5, ds, dt, additionalData)));
	}

	private static void prepareOutputDataset(
			final N5Writer n5,
			final String dataset,
			final DatasetAttributes attributes,
			final Map<String, ?> additionalAttributes) throws IOException {

		n5.createDataset(dataset, attributes);
		for (Map.Entry<String, ?> entry : additionalAttributes.entrySet())
			n5.setAttribute(dataset, entry.getKey(), entry.getValue());
	}

	private static <K, V> Map<K, V> with(Map<K, V> map, K key, V value) {

		map.put(key, value);
		return map;
	}

	private static class N5WriterSupplier implements Supplier<N5Writer>, Serializable {

		private final String container;

		private final boolean withPrettyPrinting;

		private final boolean disableHtmlEscaping;

		private final boolean serializeSpecialFloatingPointValues = true;

		private N5WriterSupplier(final String container, final boolean withPrettyPrinting, final boolean disableHtmlEscaping) {

			this.container = container;
			this.withPrettyPrinting = withPrettyPrinting;
			this.disableHtmlEscaping = disableHtmlEscaping;
		}

		@Override
		public N5Writer get() {

			final N5Factory factory = N5Helpers.defaultFactory();
			factory.gsonBuilder(createBuilder());
			return factory.openWriter(container);
		}

		private GsonBuilder createBuilder() {

			return serializeSpecialFloatingPointValues(withPrettyPrinting(disableHtmlEscaping(new GsonBuilder())));
		}

		private GsonBuilder serializeSpecialFloatingPointValues(final GsonBuilder builder) {

			return with(builder, this.serializeSpecialFloatingPointValues, GsonBuilder::serializeSpecialFloatingPointValues);
		}

		private GsonBuilder withPrettyPrinting(final GsonBuilder builder) {

			return with(builder, this.withPrettyPrinting, GsonBuilder::setPrettyPrinting);
		}

		private GsonBuilder disableHtmlEscaping(final GsonBuilder builder) {

			return with(builder, this.disableHtmlEscaping, GsonBuilder::disableHtmlEscaping);
		}

		private static GsonBuilder with(final GsonBuilder builder, boolean applyAction, Function<GsonBuilder, GsonBuilder> action) {

			return applyAction ? action.apply(builder) : builder;
		}
	}

	private static double[] ones(final int length) {

		double[] ones = new double[length];
		Arrays.fill(ones, 1.0);
		return ones;
	}

	private static Interval addDimension(final Interval interval, final long m, final long M) {

		long[] min = new long[interval.numDimensions() + 1];
		long[] max = new long[interval.numDimensions() + 1];
		for (int d = 0; d < interval.numDimensions(); ++d) {
			min[d] = interval.min(d);
			max[d] = interval.max(d);
		}
		min[interval.numDimensions()] = m;
		max[interval.numDimensions()] = M;
		return new FinalInterval(min, max);
	}

	private static String toString(final Interval interval) {

		return String.format("(%s %s)", Arrays.toString(Intervals.minAsLongArray(interval)), Arrays.toString(Intervals.maxAsLongArray(interval)));
	}

	private static double[] reversed(final double[] array, final boolean reverse) {

		return reverse ? reversed(array) : array;
	}

	private static double[] reversed(final double[] array) {

		final double[] copy = new double[array.length];
		for (int i = 0, k = copy.length - 1; i < copy.length; ++i, --k) {
			copy[i] = array[k];
		}
		return copy;
	}

	private static <T extends RealType<T>> ArrayImg<FloatType, FloatArray> smooth(
			final RandomAccessibleInterval<T> source,
			final Interval interval,
			final int channelDim,
			double sigma) {

		final ArrayImg<FloatType, FloatArray> img = ArrayImgs.floats(Intervals.dimensionsAsLongArray(interval));

		for (long channel = interval.min(channelDim); channel <= interval.max(channelDim); ++channel) {
			Gauss3.gauss(
					sigma,
					Views.extendBorder(Views.hyperSlice(source, channelDim, channel)),
					Views.hyperSlice(Views.translate(img, Intervals.minAsLongArray(interval)), channelDim, channel));
		}
		return img;
	}

	private static <T extends RealType<T>> void invalidateOutOfBlockAffinities(
			final RandomAccessibleInterval<T> affs,
			final T invalid,
			final long[]... offsets
	) {

		for (int index = 0; index < offsets.length; ++index) {
			final IntervalView<T> slice = Views.hyperSlice(affs, affs.numDimensions() - 1, index);
			for (int d = 0; d < offsets[index].length; ++d) {
				final long offset = offsets[index][d];
				if (offset == 0)
					continue;
				final long pos = offset > 0 ? slice.max(d) + 1 - offset : slice.min(d) - 1 - offset;
				Views.hyperSlice(slice, d, pos).forEach(p -> p.set(invalid));
			}
		}
	}

	private static <UF extends UnionFind> UF findOverlappingLabelsThresholdMedianEdgeAffinities(
			final JavaSparkContext sc,
			final Supplier<? extends N5Reader> labelContainer,
			final Supplier<? extends N5Reader> affinitiesContainer,
			final String labelDataset,
			final String affinitiesDataset,
			final double threshold,
			final UF uf,
			final long[] dimensions,
			final int[] blockSize,
			final int[] blocksPerTask,
			final long... ignoreThese) {

		Objects.requireNonNull(sc);
		Objects.requireNonNull(blockSize);
		Objects.requireNonNull(blocksPerTask);
		Objects.requireNonNull(ignoreThese);

		// TODO: 2/20/19

		final int[] taskBlockSize = new int[blockSize.length];
		Arrays.setAll(taskBlockSize, d -> blockSize[d] * blocksPerTask[d]);

		final List<Tuple2<long[], long[]>> doubleSizeBlocks = Grids
				.collectAllContainedIntervals(dimensions, taskBlockSize)
				.stream()
				.map(interval -> SparkWatersheds.toMinMaxTuple(interval, Tuple2::new))
				.collect(Collectors.toList());

		final List<Tuple2<long[], long[]>> mappings = sc
				.parallelize(doubleSizeBlocks)
				.mapToPair(minMax -> {
					final long[] thisBlockMax = minMax._2();

					final N5Reader labelsReader = labelContainer.get();
					final String labelsCacheKey = new URIBuilder(labelsReader.getURI()).setParameters(
							new BasicNameValuePair("call", "spark-watersheds-find-overlapping-labels"),
							new BasicNameValuePair("container", "labels"),
							new BasicNameValuePair("dataset", labelDataset)
					).toString();
					final RandomAccessibleInterval<UnsignedLongType> labels = Singleton.get(labelsCacheKey, () -> N5Helpers.openBounded(labelsReader, labelDataset));
					final RandomAccessibleInterval<UnsignedLongType> thisBlockLabels = Views.interval(labels, minMax._1(), thisBlockMax);

					final N5Reader affinitiesReader = affinitiesContainer.get();
					final String affinitiesCacheKey = new URIBuilder(affinitiesReader.getURI()).setParameters(
							new BasicNameValuePair("call", "spark-watersheds-find-overlapping-labels"),
							new BasicNameValuePair("container", "affinities"),
							new BasicNameValuePair("dataset", affinitiesDataset)
					).toString();
					final RandomAccessibleInterval<FloatType> affinities = Singleton.get(affinitiesCacheKey, () -> N5Helpers.openBounded(affinitiesContainer.get(), affinitiesDataset));

					final TLongSet ignoreTheseSet = new TLongHashSet(ignoreThese);

					final TLongLongHashMap mapping = new TLongLongHashMap();
					final UnionFind localUF = new LongHashMapUnionFind(mapping, 0, Long::compare);

					for (int dim = 0; dim < thisBlockMax.length; ++dim) {
						final long thisSliceIndex = thisBlockMax[dim];
						final long thatSliceIndex = thisSliceIndex + 1;
						if (thatSliceIndex >= dimensions[dim]) {
							LOG.debug("That slice index {} outside dimensions {}", thatSliceIndex, dimensions);
							continue;
						}

						LOG.debug("Slicing this block to {} and that block to {} for dim {}", thatSliceIndex, thatSliceIndex, dim);
						// TODO once Intervals.hyperSlice is available on imglib2 release, we do not need to use Views.hyperSlice here
						// imglib/imglib2@62998cd
						final Interval interval = Views.hyperSlice(thisBlockLabels, dim, thisBlockLabels.min(dim));
						RandomAccessibleInterval<UnsignedLongType> thisSliceLabels = Views.interval(Views.hyperSlice(labels, dim, thisSliceIndex), interval);
						RandomAccessibleInterval<UnsignedLongType> thatSliceLabels = Views.interval(Views.hyperSlice(labels, dim, thatSliceIndex), interval);
						RandomAccessibleInterval<FloatType> thisSliceAffinities = Views.interval(Views.hyperSlice(affinities, dim, thisSliceIndex), interval);
						RandomAccessibleInterval<FloatType> thatSliceAffinities = Views.interval(Views.hyperSlice(affinities, dim, thatSliceIndex), interval);
						LOG.debug("This slice: ({} {})", Intervals.minAsLongArray(thisSliceLabels), Intervals.maxAsLongArray(thisSliceLabels));
						LOG.debug("That slice: ({} {})", Intervals.minAsLongArray(thatSliceLabels), Intervals.maxAsLongArray(thatSliceLabels));

						final Cursor<UnsignedLongType> thisCursorLabels = Views.flatIterable(thisSliceLabels).cursor();
						final Cursor<UnsignedLongType> thatCursorLabels = Views.flatIterable(thatSliceLabels).cursor();

						final Cursor<FloatType> thisCursorAffinities = Views.flatIterable(thisSliceAffinities).cursor();
						final Cursor<FloatType> thatCursorAffiniities = Views.flatIterable(thatSliceAffinities).cursor();

						final TLongObjectMap<TLongObjectMap<TDoubleArrayList>> affinitiesByEdge = new TLongObjectHashMap<>();
						final LongFunction<TDoubleArrayList> emptyListFactory = key -> new TDoubleArrayList();
						final LongFunction<TLongObjectMap<TDoubleArrayList>> emptyMapFactory = key -> new TLongObjectHashMap<>();

						while (thisCursorLabels.hasNext()) {
							final long thisLabel = thisCursorLabels.next().getIntegerLong();
							final long thatLabel = thatCursorLabels.next().getIntegerLong();

							final double thisAffinity = thisCursorAffinities.next().getRealDouble();
							final double thatAffinity = thatCursorAffiniities.next().getRealDouble();

							if (ignoreTheseSet.contains(thisLabel) || ignoreTheseSet.contains(thatLabel))
								continue;

							if (thisLabel == thatLabel) {
								LOG.error(
										"Found same label {} in slices {} and {} for dimension {}",
										thisLabel,
										thisSliceIndex,
										thatSliceIndex);
								throw new RuntimeException("Got the same label in two different blocks -- impossible: " + thisLabel);
							}

							final long e1, e2;
							if (thisLabel < thatLabel) {
								e1 = thisLabel;
								e2 = thatLabel;
							} else {
								e1 = thatLabel;
								e2 = thisLabel;
							}

							if (!affinitiesByEdge.containsKey(e1))
								affinitiesByEdge.put(e1, new TLongObjectHashMap<>());

							final TLongObjectMap<TDoubleArrayList> allNeighborAffinities = computeIfAbsent(affinitiesByEdge, e1, emptyMapFactory);
							final TDoubleArrayList neighborAffinities = computeIfAbsent(allNeighborAffinities, e2, emptyListFactory);

							if (!Double.isNaN(thisAffinity))
								neighborAffinities.add(thisAffinity);

							if (!Double.isNaN(thatAffinity))
								neighborAffinities.add(thatAffinity);

						}

						LOG.info("Edge affinities: {}", affinitiesByEdge);

						affinitiesByEdge.forEachEntry((k, v) -> {
							TLongObjectIterator<TDoubleArrayList> edgeIt = v.iterator();
							while (edgeIt.hasNext()) {
								edgeIt.advance();
								double[] affinitiesSorted = edgeIt.value().toArray();
								if (affinitiesSorted.length > 0) {
									Arrays.sort(affinitiesSorted);
									if (affinitiesSorted[affinitiesSorted.length / 2] > threshold) {
										localUF.join(localUF.findRoot(k), localUF.findRoot(edgeIt.key()));
									}
								}
							}
							return true;
						});

					}

					LOG.info("Returning mapping {}", mapping);
					return new Tuple2<>(minMax, new Tuple2<>(mapping.keys(), mapping.values()));
				})
				.values()
				.collect();

		for (final Tuple2<long[], long[]> mapping : mappings) {
			final long[] keys = mapping._1();
			final long[] vals = mapping._2();
			for (int index = 0; index < keys.length; ++index) {
				final long k = keys[index];
				final long v = vals[index];
				final long r1 = uf.findRoot(k);
				final long r2 = uf.findRoot(v);
				if (r1 != r2) {
					uf.join(r1, r2);
					uf.findRoot(r1);
					uf.findRoot(r2);
				}

			}
		}

		return uf;

	}

	private static <UF extends UnionFind> UF findOverlappingLabelsArgMaxNoHalo(
			final JavaSparkContext sc,
			final Supplier<? extends N5Reader> n5,
			final String labelDataset,
			final UF uf,
			final long[] dimensions,
			final int[] blockSize,
			final int[] blocksPerTask,
			final long... ignoreThese) {

		Objects.requireNonNull(sc);
		Objects.requireNonNull(blockSize);
		Objects.requireNonNull(blocksPerTask);
		Objects.requireNonNull(ignoreThese);

		final int[] taskBlockSize = new int[blockSize.length];
		Arrays.setAll(taskBlockSize, d -> blockSize[d] * blocksPerTask[d]);

		final List<Tuple2<long[], long[]>> doubleSizeBlocks = Grids
				.collectAllContainedIntervals(dimensions, taskBlockSize)
				.stream()
				.map(interval -> SparkWatersheds.toMinMaxTuple(interval, Tuple2::new))
				.collect(Collectors.toList());

		final List<Tuple2<long[], long[]>> mappings = sc
				.parallelize(doubleSizeBlocks)
				.mapToPair(minMax -> {
					final long[] thisBlockMax = minMax._2();


					final N5Reader labelsReader = n5.get();
					final String labelsCacheKey = new URIBuilder(labelsReader.getURI()).setParameters(
							new BasicNameValuePair("call", "spark-watersheds-find-overlapping-labels-no-halo"),
							new BasicNameValuePair("container", "labels"),
							new BasicNameValuePair("dataset", labelDataset)
					).toString();
					final RandomAccessibleInterval<UnsignedLongType> labels = Singleton.get(labelsCacheKey, () -> N5Helpers.openBounded(labelsReader, labelDataset));
					final RandomAccessibleInterval<UnsignedLongType> thisBlock = Views.interval(labels, minMax._1(), thisBlockMax);

					final TLongSet ignoreTheseSet = new TLongHashSet(ignoreThese);

					final TLongLongHashMap mapping = new TLongLongHashMap();
					final UnionFind localUF = new LongHashMapUnionFind(mapping, 0, Long::compare);

					for (int dim = 0; dim < thisBlockMax.length; ++dim) {
						final long thisSliceIndex = thisBlockMax[dim];
						final long thatSliceIndex = thisSliceIndex + 1;
						if (thatSliceIndex >= dimensions[dim]) {
							LOG.debug("That slice index {} outside dimensions {}", thatSliceIndex, dimensions);
							continue;
						}

						LOG.debug("Slicing this block to {} and that block to {} for dim {}", thatSliceIndex, thatSliceIndex, dim);
						final Interval interval = Views.hyperSlice(thisBlock, dim, thisBlock.min(dim));
						RandomAccessibleInterval<UnsignedLongType> thisSlice = Views.interval(Views.hyperSlice(labels, dim, thisSliceIndex), interval);
						RandomAccessibleInterval<UnsignedLongType> thatSlice = Views.interval(Views.hyperSlice(labels, dim, thatSliceIndex), interval);
						LOG.debug("This slice: ({} {})", Intervals.minAsLongArray(thisSlice), Intervals.maxAsLongArray(thisSlice));
						LOG.debug("That slice: ({} {})", Intervals.minAsLongArray(thatSlice), Intervals.maxAsLongArray(thatSlice));

						final Cursor<UnsignedLongType> thisCursor = Views.flatIterable(thisSlice).cursor();
						final Cursor<UnsignedLongType> thatCursor = Views.flatIterable(thatSlice).cursor();

						final TLongObjectMap<TLongIntMap> thisMap = new TLongObjectHashMap<>();
						final TLongObjectMap<TLongIntMap> thatMap = new TLongObjectHashMap<>();

						while (thisCursor.hasNext()) {
							final long thisLabel = thisCursor.next().getIntegerLong();
							final long thatLabel = thatCursor.next().getIntegerLong();

							if (ignoreTheseSet.contains(thisLabel) || ignoreTheseSet.contains(thatLabel))
								continue;

							if (thisLabel == thatLabel) {
								LOG.error(
										"Found same label {} in slices {} and {} for dimension {}",
										thisLabel,
										thisSliceIndex,
										thatSliceIndex);
								throw new RuntimeException("Got the same label in two different blocks -- impossible: " + thisLabel);
							}

							if (!thisMap.containsKey(thisLabel))
								thisMap.put(thisLabel, new TLongIntHashMap());

							if (!thatMap.containsKey(thatLabel))
								thatMap.put(thatLabel, new TLongIntHashMap());

							addOne(thisMap.get(thisLabel), thatLabel);
							addOne(thatMap.get(thatLabel), thisLabel);

							//							thatArgMax.forEachEntry((k, v) -> {
							////								if (thatArgMax.get(v) == k)
							//								localUF.join(localUF.findRoot(v), localUF.findRoot(k));
							////								mapping.put(k, v);
							//								return true;
							//							});

						}

						LOG.debug("Mapping this to that {}", thisMap);
						LOG.debug("Mapping that to this {}", thatMap);

						final TLongLongMap thisArgMax = argMaxCounts(thisMap);
						final TLongLongMap thatArgMax = argMaxCounts(thatMap);

						thisArgMax.forEachEntry((k, v) -> {
							if (thatArgMax.get(v) == k)
								localUF.join(localUF.findRoot(v), localUF.findRoot(k));
							return true;
						});

					}

					LOG.debug("Returning mapping {}", mapping);
					return new Tuple2<>(minMax, new Tuple2<>(mapping.keys(), mapping.values()));
				})
				.values()
				.collect();

		for (final Tuple2<long[], long[]> mapping : mappings) {
			final long[] keys = mapping._1();
			final long[] vals = mapping._2();
			for (int index = 0; index < keys.length; ++index) {
				final long k = keys[index];
				final long v = vals[index];
				final long r1 = uf.findRoot(k);
				final long r2 = uf.findRoot(v);
				if (r1 != r2) {
					uf.join(r1, r2);
					uf.findRoot(r1);
					uf.findRoot(r2);
				}

			}
		}

		return uf;

	}

	private static class CropAffinities implements PairFunction<Interval, Interval, RandomAccessibleInterval<FloatType>> {

		private final Supplier<? extends N5Reader> n5in;

		private final String affinities;

		private final double minmimumAffinity;

		private final long[] halo;

		private CropAffinities(
				final Supplier<? extends N5Reader> n5in,
				final String affinities,
				final double minimumAffinity,
				final long[] halo) {

			this.n5in = n5in;
			this.affinities = affinities;
			this.minmimumAffinity = minimumAffinity;
			this.halo = halo;
		}

		@Override
		public Tuple2<Interval, RandomAccessibleInterval<FloatType>> call(final Interval interval) throws Exception {

			final N5Reader n5Local = n5in.get();
			final String cacheKey = new URIBuilder(n5Local.getURI()).setParameters(
					new BasicNameValuePair("affinities", affinities),
					new BasicNameValuePair("call", "spark-watersheds")
			).toString();
			final RandomAccessibleInterval<FloatType> affs = Singleton.get(cacheKey, () -> N5Helpers.openBounded(n5Local, affinities));

			final Interval withHalo = Intervals.expand(interval, halo);

			final ArrayImg<FloatType, FloatArray> affinityCrop = ArrayImgs.floats(Intervals.dimensionsAsLongArray(withHalo));
			final Cursor<FloatType> source = Views.flatIterable(Views.interval(Views.extendValue(affs, new FloatType(Float.NaN)), withHalo)).cursor();
			final Cursor<FloatType> target = Views.flatIterable(affinityCrop).cursor();
			// need to ensure that all NaN voxels are replaced with zeros
			// otherwise, LocalExtrema.MaximumCheck adds too many points and everything gets slow!
			while (source.hasNext()) {
				final double val = source.next().getRealDouble();
				target.next().setReal(Double.isNaN(val) ? Double.NEGATIVE_INFINITY : val);

			}

			return new Tuple2<>(interval, affinityCrop);
		}
	}

	private static <T> T toMinMaxTuple(final Interval interval, BiFunction<long[], long[], T> toTuple) {

		return toTuple.apply(Intervals.minAsLongArray(interval), Intervals.maxAsLongArray(interval));
	}

	private static void addOne(final TLongIntMap countMap, final long label) {

		countMap.put(label, countMap.get(label) + 1);
	}

	private static TLongLongMap argMaxCounts(final TLongObjectMap<TLongIntMap> counts) {

		final TLongLongMap mapping = new TLongLongHashMap();
		counts.forEachEntry((k, v) -> {
			mapping.put(k, argMaxCount(v));
			return true;
		});
		return mapping;
	}

	private static long argMaxCount(final TLongIntMap counts) {

		long maxCount = Long.MIN_VALUE;
		long argMaxCount = 0;
		for (final TLongIntIterator it = counts.iterator(); it.hasNext(); ) {
			it.advance();
			final long v = it.value();
			if (v > maxCount) {
				maxCount = v;
				argMaxCount = it.key();
			}
		}
		;
		return argMaxCount;
	}

	private static class CachedMapper implements LongUnaryOperator {

		private long nextId;

		private final TLongLongMap cache = new TLongLongHashMap();

		private CachedMapper(final long firstId) {

			this.nextId = firstId;
		}

		@Override
		public long applyAsLong(long l) {

			if (l == 0)
				return 0;

			if (!cache.containsKey(l))
				cache.put(l, nextId++);

			return cache.get(l);
		}
	}

	private static <T> T computeIfAbsent(final TLongObjectMap<T> map, final long key, final LongFunction<T> mappingFactory) {

		final T value = map.get(key);
		if (value != null)
			return value;
		final T t = mappingFactory.apply(key);
		map.put(key, t);
		return t;
	}

}
