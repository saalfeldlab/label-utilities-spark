package org.janelia.saalfeldlab.label.spark.affinities;

import com.google.gson.GsonBuilder;
import com.google.gson.annotations.Expose;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import kotlin.Pair;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.Point;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.algorithm.labeling.affinities.ConnectedComponents;
import net.imglib2.algorithm.labeling.affinities.Watersheds;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.algorithm.util.unionfind.IntArrayUnionFind;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.label.Label;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.MixedTransformView;
import net.imglib2.view.Views;
import org.apache.commons.lang.builder.ToStringBuilder;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.label.spark.Version;
import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5FSWriter;
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
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.function.LongUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;

public class SparkRain {

	private static final String RESOLUTION_KEY = "resolution";

	private static final String OFFSET_KEY = "offset";

	private static final String ARGUMENTS_KEY = "arguments";

	private static final String ARGV_KEY = "argumentVector";

	private static final String LABEL_UTILITIES_SPARK_KEY = "label-utilities-spark";

	private static final String VERSION_KEY = "version";

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	private static class Offset implements Serializable {

		@Expose
		private final int channelIndex;

		@Expose
		private final long[] offset;

		public Offset(final int channelIndex, final long... offset) {
			this.channelIndex = channelIndex;
			this.offset = offset;
		}

		public long[] offset() {
			return offset.clone();
		}

		public int channelIndex() {
			return channelIndex;
		}

		@Override
		public String toString() {
			return new ToStringBuilder(this)
					.append("channelIndex", channelIndex)
					.append("offset", Arrays.toString(offset))
					.toString();
		}

		public static Offset parseOffset(final String representation) {
			final String[] split = representation.split(":");
			return new Offset(
					split.length > 1 ? Integer.parseInt(split[1]) : -1,
					Stream.of(split[0].split(",")).mapToLong(Long::parseLong).toArray());
		}

		public static class Converter implements CommandLine.ITypeConverter<Offset> {

			@Override
			public Offset convert(String s) {
				return Offset.parseOffset(s);
			}
		}
	}

	private static class Args implements Serializable, Callable<Void> {

		@Expose
		@CommandLine.Parameters(arity = "1", paramLabel = "INPUT_CONTAINER", description = "Path to N5 container with affinities dataset.")
		String inputContainer = null;

		@Expose
		@CommandLine.Option(names = "--output-container", paramLabel = "OUTPUT_CONTAINER", description = "Path to output container. Defaults to INPUT_CONTAINER.")
		String outputContainer = null;

		@Expose
		@CommandLine.Option(names = "--affinity-dataset", paramLabel = "AFFINITIES", description = "Path of affinities dataset in INPUT_CONTAINER.")
		String affinities = "volumes/affinities/prediction";

		@Expose
		@CommandLine.Option(names = "--watersheds-dataset", paramLabel = "WATERSHEDS", description = "Path to watersheds in OUTPUT_CONTAINER")
		String watersheds = "volumes/labels/watersheds";

		@Expose
		@CommandLine.Option(names = "--watershed-seeds-dataset")
		String watershedSeeds = "volumes/labels/watershed_seeds";

		@Expose
		@CommandLine.Option(names = "--seeded-watersheds-dataset", paramLabel = "WATERSHEDS", description = "Path to watersheds in OUTPUT_CONTAINER")
		String seededWatersheds = "volumes/labels/seeded_watersheds";

		@Expose
		@CommandLine.Option(names = "--merged-dataset", paramLabel = "WATERSHEDS_MERGED", description = "Path to region merged in OUTPUT_CONTAINER")
		String merged = "volumes/labels/watersheds_merged";

		@Expose
		@CommandLine.Option(names = "--size-filtered-dataset", paramLabel = "SIZE_FILTERED", description = "Path to size_filtered in OUTPUT_CONTAINER (only if size filter is > 0)")
		String sizeFiltered = "volumes/labels/size_filtered";

		@Expose
		@CommandLine.Option(names = "--invert-affinities-axis", paramLabel = "INVERT_AFFINITIES_AXIS", description = "Invert axis that holds affinities. This is necessary if affinities were generated as [z,y,x].")
		Boolean invertAffinitiesAxis = false;

		@Expose
		@CommandLine.Option(names = "--threshold", paramLabel = "THRESHOLD", description = "Threshold for thresholding affinities. Defaults to 0.5.")
		Double threshold = 0.9;

		@Expose
		@CommandLine.Option(names = "--offsets", arity = "1..*", paramLabel = "OFFSETS", description = "Structuring elements for affinities. Defaults to -1,0,0 0,-1,0 0,0,-1.", converter = Offset.Converter.class)
		Offset[] offsets = {new Offset(-1, 0, 0), new Offset(0, -1, 0), new Offset(0, 0, -1)};

		@Expose
		@CommandLine.Option(names = "--block-size", paramLabel = "BLOCK_SIZE", description = "Block size of output.", split = ",")
		int[] blockSize = {64, 64, 64};

		@Expose
		@CommandLine.Option(names = "--blocks-per-task", paramLabel = "BLOCKS_PER_TASK", description = "How many blocks to combine for watersheds/connected components (one value per dimension)", split=",")
		int[] blocksPerTask = {1, 1, 1};

		@Expose
		@CommandLine.Option(names = "--halo", paramLabel = "HALO", description = "Include halo region to run connected components/watersheds", split=",")
		int[] halo = {0, 0, 0};

		@Expose
		@CommandLine.Option(names = "--size-filter", paramLabel = "MIN_SIZE", description = "Anything below this size will be considered background. Will be ignored if <= 0")
		Integer minSize = -1;

		@Expose
		@CommandLine.Option(names = "--min-watershed-affinity", paramLabel = "MIN_WATERSHED_AFFINITY", description = "Ignore edges that have lower affinity than MIN_WATERSHED_AFFINITY")
		Double minWatershedAffinity = Double.NEGATIVE_INFINITY;

		@Expose
		@CommandLine.Option(names = "--cropped-datasets-pattern", paramLabel = "CROPPED_DATASETS_PATTERN", description = "All data sets are stored with halo by default. Cropped versions according to this.", defaultValue = "%s-cropped")
		String croppedDatasetPattern;

		@Expose
		@CommandLine.Option(names = "--relabel", paramLabel = "RELABEL", description = "Relabel all label data sets for unique labels", defaultValue = "false")
		Boolean relabel;

		@Expose
		@CommandLine.Option(names = "--revert-array-attributes", paramLabel = "RELABEL", description = "Revert all array attributes (that are not dataset attributes)", defaultValue = "false")
		Boolean revertArrayAttributes;

		@Expose
		@CommandLine.Option(names = "--smooth-affinities", paramLabel = "SIGMA", description = "Smooth affinities before watersheds (if SIGMA > 0)", defaultValue = "0.0")
		Double smoothAffinitiesSigma;

		@Expose
		@CommandLine.Option(names = "--smoothed-affinities-dataset", defaultValue = "volumes/affinities/prediction_smoothed")
		String smoothedAffinities;

		@CommandLine.Option(names = "--json-pretty-print", defaultValue = "true")
		transient Boolean prettyPrint;

		@CommandLine.Option(names = "--json-disable-html-escape", defaultValue = "true")
		transient Boolean disbaleHtmlEscape;

		@CommandLine.Option(names = { "-h", "--help"}, usageHelp = true, description = "Display this help and exit")
		private Boolean help;

		@Override
		public Void call() throws Exception {

			final long numSpecifiedOffsetChannelIndices = Stream.of(offsets).filter(o -> o.channelIndex >= 0).count();

			if (numSpecifiedOffsetChannelIndices == 0) {
				final DatasetAttributes attributes = new N5FSWriter(inputContainer).getDatasetAttributes(affinities);
				if (attributes.getDimensions()[attributes.getNumDimensions() - 1] != this.offsets.length)
					throw new Exception("Need to define all offsets when not specifying channel indices explicitly.");
			} else if (numSpecifiedOffsetChannelIndices < offsets.length - 1)
				throw new Exception("Can only specify all or no channel indices for offsets.");

			return null;
		}

		public Offset[] enumeratedOffsets() {
			final Offset[] enumeratedOffsets = new Offset[this.offsets.length];
			for (int i = 0; i < offsets.length; ++i) {
				final Offset o = this.offsets[i];
				enumeratedOffsets[i] = new Offset(
						o.channelIndex == -1 ? i : o.channelIndex,
						o.offset.clone());
			}
			return enumeratedOffsets;
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

		final DatasetAttributes inputAttributes = n5in.get().getDatasetAttributes(args.affinities);
		final long[] inputDims = inputAttributes.getDimensions();
		final long[] outputDims = new long[inputDims.length - 1];
		Arrays.setAll(outputDims, d -> inputDims[d]);

		final Map<String, Object> labelUtilitiesSparkAttributes = new HashMap<>();
		labelUtilitiesSparkAttributes.put(ARGUMENTS_KEY, args);
		labelUtilitiesSparkAttributes.put(ARGV_KEY, argv);
		labelUtilitiesSparkAttributes.put(VERSION_KEY, Version.VERSION_STRING);
		final Map<String, Object> attributes = with(new HashMap<>(), LABEL_UTILITIES_SPARK_KEY, labelUtilitiesSparkAttributes);


		final int[] taskBlockSize = IntStream.range(0, args.blockSize.length).map(d -> args.blockSize[d] * args.blocksPerTask[d]).toArray();
		final boolean hasHalo = Arrays.stream(args.halo).filter(h -> h != 0).count() > 0;
		if (hasHalo)
			throw new UnsupportedOperationException("Halo currently not supported, please omit halo option!");

		String[] uint64Datasets = args.minSize > 0
				? new String[] {args.watersheds, args.merged, args.seededWatersheds, args.sizeFiltered}
				: new String[] {args.watersheds, args.merged, args.seededWatersheds};
		String[] uint8Datasets = {args.watershedSeeds};

		String[] float32Datasets = args.smoothAffinitiesSigma > 0 ? new String[] {args.smoothedAffinities} : new String[] {};

		final double[] resolution = reverted(Optional.ofNullable(n5in.get().getAttribute(args.affinities, RESOLUTION_KEY, double[].class)).orElse(ones(outputDims.length)), args.revertArrayAttributes);
		final double[] offset = reverted(Optional.ofNullable(n5in.get().getAttribute(args.affinities, OFFSET_KEY, double[].class)).orElse(new double[outputDims.length]), args.revertArrayAttributes);
		attributes.put(RESOLUTION_KEY, resolution);
		attributes.put(OFFSET_KEY, offset);

		final Map<String, DatasetAttributes> datasets = new HashMap<>();
		Arrays.asList(uint64Datasets).forEach(ds -> datasets.put(ds, new DatasetAttributes(outputDims, args.blockSize, DataType.UINT64, new GzipCompression())));
		Arrays.asList(uint8Datasets).forEach(ds -> datasets.put(ds, new DatasetAttributes(outputDims, args.blockSize, DataType.UINT8, new GzipCompression())));

		if (args.smoothAffinitiesSigma > 0.0)
			prepareOutputDataset(n5out.get(), args.smoothedAffinities, new DatasetAttributes(inputDims, IntStream.concat(IntStream.of(args.blockSize), IntStream.of(1)).toArray(), DataType.FLOAT32, new GzipCompression()), attributes);

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

		final Offset[] offsets = args.enumeratedOffsets();


		final SparkConf conf = new SparkConf().setAppName(MethodHandles.lookup().lookupClass().getName());
		try (final JavaSparkContext sc = new JavaSparkContext(conf)) {
			run(
					sc,
					n5in,
					n5out,
					outputDims,
					IntStream.of(args.halo).mapToLong(i -> i).toArray(),
					args.invertAffinitiesAxis,
					args.minWatershedAffinity,
					args.threshold,
					args.minSize,
					Stream.of(offsets).map(Offset::offset).toArray(long[][]::new),
					Stream.of(offsets).mapToInt(Offset::channelIndex).toArray(),
					args.smoothAffinitiesSigma,
					args.affinities,
					args.watersheds,
					args.merged,
					args.sizeFiltered,
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
			final N5WriterSupplier n5in,
			final N5WriterSupplier n5out,
			final long[] outputDims,
			final long[] halo,
			final boolean invertAffinitiesAxis,
			final double minWatershedAffinity,
			final double threshold,
			final int minSize,
			final long[][] offsets,
			final int[] offsetChannelIndices,
			final double smoothAffinitiesSigma,
			final String affinities,
			final String watersheds,
			final String merged,
			final String sizeFiltered,
			final String watershedSeeds,
			final String seededWatersheds,
			final String croppedDatasetPattern,
			final int[] blockSize,
			final int[] blocksPerTask,
			final boolean relabel) throws IOException {

		final int numChannels = offsets.length;
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
				.map(t -> new FinalInterval(t._1(), t._2()))
				.mapToPair(block -> new Tuple2<>(block, N5Utils.<FloatType>open(n5in.get(), affinities)))
				.mapValues(affs -> invertAffinitiesAxis ? Views.zeroMin(Views.invertAxis(affs, affs.numDimensions() - 1)) : affs)
				.mapValues(affs -> Views.stack(IntStream.of(offsetChannelIndices).mapToObj(idx -> Views.hyperSlice(affs, affs.numDimensions() - 1, (long) idx)).collect(Collectors.toList())))
				.mapToPair(t -> {
					final Interval withHalo = Intervals.expand(t._1(), halo);
					final Interval withHaloAndChannels = addDimension(withHalo, 0, offsets.length - 1);
					final ArrayImg<FloatType, FloatArray> affinityCrop = ArrayImgs.floats(Intervals.dimensionsAsLongArray(withHaloAndChannels));
					LoopBuilder.setImages(affinityCrop, Views.interval(Views.extendValue(t._2(), new FloatType(Float.NaN)), withHaloAndChannels)).forEachPixel(FloatType::set);
					return smoothAffinitiesSigma > 0.0
						? new Tuple2<>(t._1(), new Tuple2<>(affinityCrop, smooth(t._2(), withHaloAndChannels, withHaloAndChannels.numDimensions() - 1, smoothAffinitiesSigma)))
						: new Tuple2<>(t._1(), new Tuple2<>(affinityCrop, (ArrayImg<FloatType, FloatArray>) null));
				})
				.mapValues(affs -> {
					// TODO how to avoid looking outside interval?
					// TODO optimize this!
					invalidateOutOfBlockAffinities(affs._1(), new FloatType(Float.NaN), offsets);
					invalidateOutOfBlockAffinities(affs._2(), new FloatType(Float.NaN), offsets);
					return affs;
				})
				.mapToPair(t -> {
					final Interval block = t._1();
					final RandomAccessibleInterval<FloatType> uncollapsedAffinities = t._2()._1();
					final RandomAccessibleInterval<FloatType> uncollapsedSmoothedAffinities = t._2()._2();

					final CellGrid grid = new CellGrid(outputDims, blockSize);
					final CellGrid watershedsGrid = new CellGrid(outputDims, watershedBlockSize);
					LOG.debug("Got grids {} and {}", grid, watershedsGrid);

					final long[] blockOffset = Intervals.minAsLongArray(block);
					final long[] watershedsBlockOffset = blockOffset.clone();
					grid.getCellPosition(blockOffset, blockOffset);
					watershedsGrid.getCellPosition(watershedsBlockOffset, watershedsBlockOffset);
					LOG.debug("min={} blockOffset={} watershedsBlockOffset={}", Intervals.minAsLongArray(block), blockOffset, watershedsBlockOffset);

					final int[] symmetricOrder = new int[offsets.length];
					Arrays.setAll(symmetricOrder, d -> offsets.length - 1 - d);
					final RandomAccessibleInterval<FloatType> symmetricAffinities = Watersheds.constructAffinities(
							uncollapsedAffinities,
							offsets,
							new ArrayImgFactory<>(new FloatType()),
							symmetricOrder
					);

					final RandomAccessibleInterval<FloatType> symmetricSmoothedAffinities = uncollapsedSmoothedAffinities == null
							? symmetricAffinities
							: Watersheds.constructAffinities(
									uncollapsedSmoothedAffinities,
									offsets,
									new ArrayImgFactory<>(new FloatType()),
									symmetricOrder);
//					}
					final long[][] symmetricOffsets = Watersheds.symmetricOffsets(Watersheds.SymmetricOffsetOrder.ABCCBA, offsets);

					final Pair<long[], long[]> parentsAndRoots = Watersheds.letItRain(
							Views.collapseReal(symmetricAffinities),
							minWatershedAffinity > Double.NEGATIVE_INFINITY ? v -> !Double.isNaN(v.getRealDouble()) && v.getRealDouble() >= minWatershedAffinity : v -> !Double.isNaN(v.getRealDouble()),
							(c, u) -> c.getRealDouble() > u.getRealDouble(),
							new FloatType(Float.NEGATIVE_INFINITY),
							symmetricOffsets
					);

					final long[] parents = parentsAndRoots.getFirst();
					final long[] roots = parentsAndRoots.getSecond();

					final long[] dims = Intervals.dimensionsAsLongArray(Views.collapseReal(symmetricAffinities));
					final ArrayImg<UnsignedLongType, LongArray> labels = ArrayImgs.unsignedLongs(parentsAndRoots.getFirst(), dims);
					final Interval relevantInterval = Intervals.expand(labels, negativeHalo);

					final DatasetAttributes croppedAttributes = new DatasetAttributes(outputDims, blockSize, DataType.UINT64, new GzipCompression());
					final DatasetAttributes watershedAttributes = new DatasetAttributes(outputDims, watershedBlockSize, DataType.UINT64, new GzipCompression());

					LOG.debug("Saving cropped watersheds to {}", hasHalo ? String.format(croppedDatasetPattern, watersheds) : watersheds);
					N5Utils.saveBlock(Views.interval(labels, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, watersheds) : watersheds, croppedAttributes, blockOffset);
					if (hasHalo) {
						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(labels), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
						LOG.debug("Saving watershed block with halo to {}", watersheds);
						n5out.get().writeBlock(watersheds, watershedAttributes, dataBlock);
					}

					final ArrayImg<BitType, LongArray> um = ArrayImgs.bits(dims);
					final IntArrayUnionFind uf = new IntArrayUnionFind(roots.length);

					final RandomAccessibleInterval<BitType> mask = Converters.convert((RandomAccessibleInterval<UnsignedLongType>) labels, (s, tgt) -> tgt.set(s.getIntegerLong() > 0), new BitType());
					final ConnectedComponents.ToIndex toIndex = (it, index) -> parents[(int) index];
					ConnectedComponents.unionFindFromSymmetricAffinities(
							Views.extendValue(mask, new BitType(false)),
							Views.collapseReal(uncollapsedAffinities),
							Views.extendValue(um, new BitType(false)),
							uf,
							threshold,
							offsets,
							toIndex);
					Views.flatIterable(labels).forEach(vx -> vx.set(uf.findRoot(vx.getIntegerLong())));

					N5Utils.saveBlock(Views.interval(labels, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, merged) : merged, croppedAttributes, blockOffset);
					if (hasHalo) {
						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(labels), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
						n5out.get().writeBlock(merged, watershedAttributes, dataBlock);
					}


					final TIntIntHashMap counts = new TIntIntHashMap();
					for (final UnsignedLongType vx : Views.iterable(labels)) {
						final int v = vx.getInteger();
						counts.put(v, counts.get(v) + 1);
					}

					LOG.debug("Got counts: {}", counts);

					if (minSize > 0) {
						final TIntSet tooSmall = new TIntHashSet();
						counts.forEachEntry((key, value) -> {
							if (value < minSize)
								tooSmall.add(key);
							return true;
						});

						for (final UnsignedLongType vx : Views.iterable(labels)) {
							if (tooSmall.contains(vx.getInteger()))
								vx.setInteger(0);
						}
						final RandomAccessibleInterval<UnsignedLongType> relevantLabels = Views.zeroMin(Views.interval(labels, relevantInterval));
						N5Utils.saveBlock(relevantLabels, n5out.get(), hasHalo ? String.format(croppedDatasetPattern, sizeFiltered) : sizeFiltered, croppedAttributes, blockOffset);
						if (hasHalo) {
							final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(labels), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
							n5out.get().writeBlock(sizeFiltered, watershedAttributes, dataBlock);
						}
					}

					// TODO do seeded watersheds
					final RandomAccessibleInterval<BitType> watershedSeedsMaskImg = ArrayImgs.bits(Intervals.dimensionsAsLongArray(labels));
					Watersheds.seedsFromMask(Views.extendValue(labels, new UnsignedLongType(Label.OUTSIDE)), watershedSeedsMaskImg, symmetricOffsets);
					final List<Point> seeds = Watersheds.collectSeeds(watershedSeedsMaskImg);
					LOG.debug("Found watershed seeds {}", seeds);
					final RandomAccessibleInterval<UnsignedByteType> watershedSeedsMaskImgUint8 = Converters.convert(watershedSeedsMaskImg, (src,tgt) -> tgt.set(src.get() ? 1 : 0), new UnsignedByteType());
					final DatasetAttributes croppedWatershedSeedsAtributes = new DatasetAttributes(outputDims, blockSize, DataType.UINT8, new GzipCompression());
					N5Utils.saveBlock(Views.interval(watershedSeedsMaskImgUint8, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, watershedSeeds) : watershedSeeds, croppedWatershedSeedsAtributes, blockOffset);
					if (hasHalo) {
						throw new UnsupportedOperationException("Need to implement halo support!");
//						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(watershedSeedsMaskImg), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
//						n5out.get().writeBlock(watershedSeeds, watershedAttributes, dataBlock);
					}

					LOG.debug("Starting seeded watersheds with offsets {}", (Object) symmetricOffsets);
					Watersheds.seededFromAffinities(
							Views.collapseReal(symmetricSmoothedAffinities),
							labels,
							seeds,
							symmetricOffsets,
							new UnsignedLongType(0L),
							aff -> !Double.isNaN(aff));

					N5Utils.saveBlock(Views.interval(labels, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, croppedAttributes, blockOffset);
					if (hasHalo) {
						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(labels), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
						n5out.get().writeBlock(seededWatersheds, watershedAttributes, dataBlock);
					}

					return new Tuple2<>(new Tuple2<>(Intervals.minAsLongArray(t._1()), Intervals.maxAsLongArray(t._1())), roots.length - 1);
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
			sc
					.parallelizePairs(idOffsets)
					.map(t -> {
						final N5Writer n5 = n5out.get();
						final Interval interval = new FinalInterval(t._1()._1(), t._1()._2());
						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, watersheds) : watersheds, interval, t._2());
						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, merged) : merged, interval, t._2());
						relabel(n5, hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, interval, t._2());
						if (minSize > 0)
							relabel(n5, hasHalo ? String.format(croppedDatasetPattern, sizeFiltered) : sizeFiltered, interval, t._2());

						if (hasHalo)
							throw new UnsupportedOperationException("Halo relabeling not implemented yet!");

						// TODO do halo relabeling

						return true;
					})
					.count();
			final long maxId = startIndex;
			final N5Writer n5 = n5out.get();
			n5.setAttribute(hasHalo ? String.format(croppedDatasetPattern, watersheds) : watersheds, "maxId", maxId);
			n5.setAttribute(hasHalo ? String.format(croppedDatasetPattern, merged) : merged, "maxId", maxId);
			n5.setAttribute(hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, "maxId", maxId);
			if (minSize > 0)
				n5.setAttribute(hasHalo ? String.format(croppedDatasetPattern, sizeFiltered) : sizeFiltered, "maxId", maxId);

			if (hasHalo)
				throw new UnsupportedOperationException("Halo relabeling not implemented yet!");
		}

	}

	private static void relabel(
			final N5Writer n5,
			final String dataset,
			final Interval interval,
			final long addIfNotZero) throws IOException {
		SparkRain.<UnsignedLongType>relabel(n5, dataset, interval, (src, tgt) -> {
			final long val = src.getIntegerLong();
			tgt.set(val == 0 ? 0 : val + addIfNotZero);
		});
	}

	private static <T extends IntegerType<T> & NativeType<T>> void relabel(
			final N5Writer n5,
			final String dataset,
			final Interval interval,
			final BiConsumer<T, T> idMapping) throws IOException {
		final DatasetAttributes attributes = n5.getDatasetAttributes(dataset);
		final CellGrid grid = new CellGrid(attributes.getDimensions(), attributes.getBlockSize());
		final RandomAccessibleInterval<T> data = Views.interval(N5Utils.<T>open(n5, dataset), interval);
		final RandomAccessibleInterval<T> copy = new ArrayImgFactory<>(Util.getTypeFromInterval(data).createVariable()).create(data);
		LoopBuilder.setImages(data, copy).forEachPixel(idMapping);
		final long[] blockPos = Intervals.minAsLongArray(interval);
		grid.getCellPosition(blockPos, blockPos);
		N5Utils.saveBlock(copy, n5, dataset, attributes, blockPos);
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

}
