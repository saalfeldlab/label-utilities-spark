package org.janelia.saalfeldlab.label.spark;

import com.google.gson.GsonBuilder;
import com.google.gson.annotations.Expose;
import gnu.trove.iterator.TLongIntIterator;
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
import net.imglib2.algorithm.util.unionfind.IntArrayUnionFind;
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
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import pl.touk.throwing.ThrowingBiConsumer;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Paths;
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
		@CommandLine.Option(names = "--watershed-seeds-dataset")
		String watershedSeeds = "volumes/labels/watershed_seeds";

		@Expose
		@CommandLine.Option(names = "--seeded-watersheds-dataset", paramLabel = "WATERSHEDS", description = "Path to watersheds in OUTPUT_CONTAINER")
		String seededWatersheds = "volumes/labels/seeded_watersheds";

		@Expose
		@CommandLine.Option(names = "--merged-dataset", paramLabel = "WATERSHEDS_MERGED", description = "Path to region merged in OUTPUT_CONTAINER")
		String merged = "volumes/labels/seeded_watersheds_merged";

		@Expose
		@CommandLine.Option(names = "--block-merged-dataset", paramLabel = "WATERSHEDS_MERGED", description = "Path to region merged in OUTPUT_CONTAINER")
		String blockMerged = "volumes/labels/seeded_watersheds_block_merged";

		@Expose
		@CommandLine.Option(names = "--block-size", paramLabel = "BLOCK_SIZE", description = "Block size of output.", split = ",")
		int[] blockSize = {64, 64, 64};

		@Expose
		@CommandLine.Option(names = "--blocks-per-task", paramLabel = "BLOCKS_PER_TASK", description = "How many blocks to combine for watersheds/connected components (one value per dimension)", split=",")
		int[] blocksPerTask = {1, 1, 1};

		@Expose
		@CommandLine.Option(names = "--threshold", paramLabel = "THRESHOLD", description = "Threshold for thresholding affinities.", defaultValue = "0.9")
		Double threshold;

		@Expose
		@CommandLine.Option(names = "--minimum-watershed-affinity", paramLabel = "THRESHOLD", description = "Threshold for thresholding affinities.")
		Double minimumAffinity = Double.NEGATIVE_INFINITY;

		@Expose
		@CommandLine.Option(names = "--halo", paramLabel = "HALO", description = "Include halo region to run connected components/watersheds", split=",")
		int[] halo = {0, 0, 0};

		@Expose
		@CommandLine.Option(names = "--cropped-datasets-pattern", paramLabel = "CROPPED_DATASETS_PATTERN", description = "All data sets are stored with halo by default. Cropped versions according to this.", defaultValue = "%s-cropped")
		String croppedDatasetPattern;

		@Expose
		@CommandLine.Option(names = "--relabel", paramLabel = "RELABEL", description = "Relabel all label data sets for unique labels", defaultValue = "false")
		Boolean relabel;

		@Expose
		@CommandLine.Option(names = "--revert-array-attributes", paramLabel = "RELABEL", description = "Revert all array attributes (that are not dataset attributes)", defaultValue = "false")
		Boolean revertArrayAttributes;

		@CommandLine.Option(names = "--json-pretty-print", defaultValue = "true")
		transient Boolean prettyPrint;

		@CommandLine.Option(names = "--json-disable-html-escape", defaultValue = "true")
		transient Boolean disbaleHtmlEscape;

		@CommandLine.Option(names = { "-h", "--help"}, usageHelp = true, description = "Display this help and exit")
		private Boolean help;

		@Override
		public Void call() {

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

		final double[] resolution = reverted(Optional.ofNullable(n5in.get().getAttribute(args.averagedAffinities, RESOLUTION_KEY, double[].class)).orElse(ones(outputDims.length)), args.revertArrayAttributes);
		final double[] offset = reverted(Optional.ofNullable(n5in.get().getAttribute(args.averagedAffinities, OFFSET_KEY, double[].class)).orElse(new double[outputDims.length]), args.revertArrayAttributes);
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
					args.threshold,
					args.minimumAffinity,
					IntStream.of(args.halo).mapToLong(i -> i).toArray(),
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
			final double mergeThreshold,
			final double minimumWatershedAffinity,
			final long[] halo,
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
				.map(t -> (Interval) new FinalInterval(t._1(), t._2()))
				.mapToPair(new CropAffinities(n5in, averagedAffinities, minimumWatershedAffinity, halo))
				.mapToPair(t -> {
					final Interval block = t._1();
					final RandomAccessibleInterval<FloatType> relief = t._2();
					List<Point> seeds = LocalExtrema
							.findLocalExtrema(
									Views.extendValue(relief, new FloatType(Float.MIN_VALUE)),
									relief,
									new LocalExtrema.MaximumCheck<>(new FloatType((float)minimumWatershedAffinity)));

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

					final IntArrayUnionFind uf = new IntArrayUnionFind(seeds.size() + 1);
					for (int d = 0; d < relief.numDimensions(); ++d) {
						final long[] min1 = Intervals.minAsLongArray(relief);
						final long[] max1 = Intervals.maxAsLongArray(relief);
						final long[] min2 = min1.clone();
						final long[] max2 = max1.clone();
						max1[d] -= 1;
						min2[d] += 1;
						final Cursor<FloatType> reliefCursor1 = Views.flatIterable(Views.interval(relief, min1, max1)).cursor();
						final Cursor<FloatType> reliefCursor2 = Views.flatIterable(Views.interval(relief, min2, max2)).cursor();
						final Cursor<UnsignedLongType> labelsCursor1 = Views.flatIterable(Views.interval(labels, min1, max1)).cursor();
						final Cursor<UnsignedLongType> labelsCursor2 = Views.flatIterable(Views.interval(labels, min2, max2)).cursor();
						while (reliefCursor1.hasNext()) {
							reliefCursor1.fwd();
							reliefCursor2.fwd();
							labelsCursor1.fwd();
							labelsCursor2.fwd();
							if (reliefCursor1.get().getRealDouble() > mergeThreshold && reliefCursor2.get().getRealDouble() > mergeThreshold){
								final long r1 = uf.findRoot(labelsCursor1.get().getIntegerLong());
								final long r2 = uf.findRoot(labelsCursor2.get().getIntegerLong());
								if (r1 != r2 && r1 != 0 && r2 != 0)
									uf.join(r1, r2);
							}
						}
					}

					// TODO find better logic instead of label == 0 check
					final TLongSet ids = new TLongHashSet();
					final TLongIntHashMap counts = new TLongIntHashMap();
					for (final UnsignedLongType pix : labels) {
						final long label = pix.getIntegerLong();
						if (label == 0)
							counts.put(label, counts.get(label) + 1);
						else {
							final long root = uf.findRoot(pix.getIntegerLong());
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
				.collect()
				;

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
			n5out.get().setAttribute(hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, "maxId", maxId);
			n5out.get().setAttribute(hasHalo ? String.format(croppedDatasetPattern, merged) : merged, "maxId", maxId);
			n5out.get().setAttribute(hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, "maxId", maxId);

			if (hasHalo)
				throw new UnsupportedOperationException("Halo relabeling not implemented yet!");

			if (maxId + 2 > Integer.MAX_VALUE)
				throw new RuntimeException("Currently only Integer.MAX_VALUE labels supported");

			final IntArrayUnionFind uf = findOverlappingLabelsArgMaxNoHalo(sc, n5out, merged, new IntArrayUnionFind((int) (maxId + 2)), outputDims, blockSize, blocksPerTask, 0);
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
				final long[] keys = new long[(int) (maxLabel - minLabel)];
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
		final DatasetAttributes attributes = n5.getDatasetAttributes(source);
		final CellGrid grid = new CellGrid(attributes.getDimensions(), attributes.getBlockSize());
		final RandomAccessibleInterval<T> data = Views.interval(N5Utils.<T>open(n5, source), interval);
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
		N5Utils.saveBlock(copy, n5, target, attributes, blockPos);
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
		final LongArrayDataBlock block = ((LongArrayDataBlock) n5.readBlock(dataset, attributes, blockPos));
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

			try {
				return Files.isDirectory(Paths.get(container))
						? new N5FSWriter(container, createaBuilder())
						: new N5HDF5Writer(container);
			} catch (final IOException e) {
				throw new RuntimeException(e);
			}
		}

		private GsonBuilder createaBuilder() {
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

	private static double[] reverted(final double[] array, final boolean revert) {
		return revert ? reverted(array) : array;
	}

	private static double[] reverted(final double[] array) {
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

					final RandomAccessibleInterval<UnsignedLongType> labels = N5Utils.open(n5.get(), labelDataset);
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
			final RandomAccessibleInterval<FloatType> affs = N5Utils.open(n5in.get(), affinities);

			final Interval withHalo = Intervals.expand(interval, halo);

			final ArrayImg<FloatType, FloatArray> affinityCrop = ArrayImgs.floats(Intervals.dimensionsAsLongArray(withHalo));
			final Cursor<FloatType> source = Views.flatIterable(Views.interval(Views.extendValue(affs, new FloatType(Float.NaN)), withHalo)).cursor();
			final Cursor<FloatType> target = Views.flatIterable(affinityCrop).cursor();

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
		};
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

}
