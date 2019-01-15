package org.janelia.saalfeldlab.label.spark.affinities;

import com.google.gson.annotations.Expose;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import kotlin.Pair;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
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
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
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
import pl.touk.throwing.ThrowingConsumer;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class SparkRain {

	private static final String RESOLUTION_KEY = "resolution";

	private static final String OFFSET_KEY = "offset";

	private static final String ARGUMENTS_KEY = "arguments";

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	private static class Offset {

		@Expose
		private final long[] offset;

		public Offset(final long... offset) {
			this.offset = offset;
		}

		public long[] offset() {
			return offset.clone();
		}

		@Override
		public String toString() {
			return super.toString();
		}
	}

	private static class Args {

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
		Double threshold = 0.5;

		@Expose
		@CommandLine.Option(names = "--offsets", arity = "1..*", paramLabel = "OFFSETS", description = "Structuring elements for affinities. Defaults to -1,0,0 0,-1,0 0,0,-1.")
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

	}

	public static void main(final String[] argv) throws IOException {

		run(argv);

	}

	public static void run(final String... argv) throws IOException {

		final Args args = new Args();
		final CommandLine cmdLine = new CommandLine(args)
				.registerConverter(Offset.class, it -> new Offset(Stream.of(it.split(",")).mapToLong(Long::parseLong).toArray()));
		cmdLine.parse(argv);

		final N5WriterSupplier n5in = new N5WriterSupplier(args.inputContainer);

		final N5WriterSupplier n5out = args.outputContainer == null
				? n5in
				: new N5WriterSupplier(args.outputContainer);

		final DatasetAttributes inputAttributes = n5in.get().getDatasetAttributes(args.affinities);
		final long[] inputDims = inputAttributes.getDimensions();
		final long[] outputDims = new long[inputDims.length - 1];
		Arrays.setAll(outputDims, d -> inputDims[d]);

		if (inputDims[inputDims.length - 1] != args.offsets.length)
			throw new RuntimeException(String.format(
					"Offsets and input dimensions inconsistent: %s -- %s",
					Arrays.toString(args.offsets),
					Arrays.toString(inputDims)));

		final Map<String, Object> attributes = new HashMap<String, Object>();
		attributes.put(ARGUMENTS_KEY, args);

		final int[] taskBlockSize = IntStream.range(0, args.blockSize.length).map(d -> args.blockSize[d] * args.blocksPerTask[d]).toArray();
		final boolean hasHalo = Arrays.stream(args.halo).filter(h -> h != 0).count() > 0;
		if (hasHalo)
			throw new RuntimeException("Halo currently not supported, please omit halo option!");

		if (hasHalo)
			prepareOutputDatasets(
					n5out.get(),
					DataType.UINT64,
					outputDims,
					taskBlockSize,
					Optional.ofNullable(n5in.get().getAttribute(args.affinities, RESOLUTION_KEY, double[].class)).orElse(ones(outputDims.length)),
					Optional.ofNullable(n5in.get().getAttribute(args.affinities, OFFSET_KEY, double[].class)).orElse(new double[outputDims.length]),
					attributes,
					args.watersheds,
					args.merged,
					args.sizeFiltered,
					args.seededWatersheds);

		prepareOutputDatasets(
				n5out.get(),
				DataType.UINT64,
				outputDims,
				args.blockSize,
				Optional.ofNullable(n5in.get().getAttribute(args.affinities, RESOLUTION_KEY, double[].class)).orElse(ones(outputDims.length)),
				Optional.ofNullable(n5in.get().getAttribute(args.affinities, OFFSET_KEY, double[].class)).orElse(new double[outputDims.length]),
				attributes,
				Stream.of(args.watersheds, args.merged, args.sizeFiltered, args.seededWatersheds).map(ds -> hasHalo ? String.format(args.croppedDatasetPattern, ds) : ds ).toArray(String[]::new));


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
					Stream.of(args.offsets).map(Offset::offset).toArray(long[][]::new),
					args.affinities,
					args.watersheds,
					args.merged,
					args.sizeFiltered,
					args.seededWatersheds,
					args.croppedDatasetPattern,
					args.blockSize,
					args.blocksPerTask);
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
			final String affinities,
			final String watersheds,
			final String merged,
			final String sizeFiltered,
			final String seededWatersheds,
			final String croppedDatasetPattern,
			final int[] blockSize,
			final int[] blocksPerTask) {

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

		sc
				.parallelize(watershedBlocks)
				.map(t -> new FinalInterval(t._1(), t._2()))
				.mapToPair(block -> new Tuple2<>(block, N5Utils.<FloatType>open(n5in.get(), affinities)))
				.mapValues(affs -> invertAffinitiesAxis ? Views.zeroMin(Views.invertAxis(affs, affs.numDimensions() - 1)) : affs)
				.mapToPair(t -> {
					final Interval withHalo = Intervals.expand(t._1(), halo);
					final Interval withHaloAndChannels = addDimension(withHalo, 0, offsets.length);
					final ArrayImg<FloatType, FloatArray> affinityCrop = ArrayImgs.floats(Intervals.dimensionsAsLongArray(withHaloAndChannels));
					LoopBuilder.setImages(affinityCrop, Views.interval(Views.extendValue(t._2(), new FloatType(Float.NaN)), withHaloAndChannels)).forEachPixel(FloatType::set);
					return new Tuple2<>(t._1(), affinityCrop);
				})
				.mapValues(affs -> {
					// TODO how to avoid looking outside interval?
					// TODO optimize this!
					for (int index = 0; index < offsets.length; ++index) {
						final IntervalView<FloatType> slice = Views.hyperSlice(affs, affs.numDimensions() - 1, index);
						for (int d = 0; d < offsets[index].length; ++d) {
							final long offset = offsets[index][d];
							if (offset == 0)
								continue;
							final long pos = offset > 0 ? slice.max(d) + 1 - offset : slice.min(d) - 1 - offset;
							Views.hyperSlice(slice, d, pos).forEach( p -> p.setReal(Float.NaN));
						}
					}
					return affs;
				})
				.mapToPair(t -> {
					final Interval block = t._1();
					final RandomAccessibleInterval<FloatType> uncollapsedAffinities = t._2();

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

					final long[][] symmetricOffsets = new long[offsets.length*2][];
					for (int index = 0; index < offsets.length; ++index) {
						symmetricOffsets[index] = offsets[index].clone();
						symmetricOffsets[index + offsets.length] = offsets[offsets.length - 1 - index].clone();
						for (int d = 0; d < symmetricOffsets[index + offsets.length].length; ++d)
							symmetricOffsets[index + offsets.length][d] *= -1;
					}

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

					final RandomAccessibleInterval<BitType> mask = Converters.convert((RandomAccessibleInterval<UnsignedLongType>)labels, (s, tgt) -> tgt.set(s.getIntegerLong() > 0), new BitType());
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
//					final RandomAccessibleInterval<BitType> watershedSeedsMaskImg = ArrayImgs.bits(Intervals.dimensionsAsLongArray(labels));
//					final RandomAccessibleInterval<BitType> foreground = Converters.convert((RandomAccessibleInterval<UnsignedLongType>)labels, (src, tgt) -> tgt.set(src.getIntegerLong() != 0), new BitType());
//					Watersheds.seedsFromMask(Views.extendValue(foreground, new BitType(true)), watershedSeedsMaskImg, Watersheds.symmetricOffsets(offsets));
//					final List<Point> seeds = Watersheds.collectSeeds(watershedSeedsMaskImg);
//					LOG.debug("Found watershed seeds {}", seeds);
//
//					Watersheds.seededFromAffinities(
//							Views.collapseReal(symmetricAffinities),
//							labels,
//							seeds,
//							offsets,
//							new UnsignedLongType(0L));
//
//					N5Utils.saveBlock(Views.interval(labels, relevantInterval), n5out.get(), hasHalo ? String.format(croppedDatasetPattern, seededWatersheds) : seededWatersheds, croppedAttributes, blockOffset);
//					if (hasHalo) {
//						final DataBlock<long[]> dataBlock = new LongArrayDataBlock(Intervals.dimensionsAsIntArray(labels), watershedsBlockOffset, labels.update(null).getCurrentStorageArray());
//						n5out.get().writeBlock(seededWatersheds, watershedAttributes, dataBlock);
//					}

					return new Tuple2<>(t._1(), parentsAndRoots);
				})
				.count();
		;

	}

	private static void prepareOutputDatasets(
			final N5Writer n5,
			final DataType dataType,
			final long[] dims,
			final int[] blockSize,
			final double[] resolution,
			final double[] offset,
			final Map<String, Object> additionalData,
			final String... datasets
	) throws IOException {

		additionalData.put(RESOLUTION_KEY, resolution);
		additionalData.put(OFFSET_KEY, offset);
		Arrays.asList(datasets).forEach(ThrowingConsumer.unchecked(ds -> prepareOutputDataset(n5, ds, dims ,blockSize, dataType, additionalData)));
	}

	private static void prepareOutputDataset(
			final N5Writer n5,
			final String dataset,
			final long[] dims,
			final int[] blockSize,
			final DataType dataType,
			final Map<String, ?> additionalAttributes) throws IOException {
		n5.createDataset(dataset, dims, blockSize, dataType, new GzipCompression());
		for (Map.Entry<String, ?> entry : additionalAttributes.entrySet())
			n5.setAttribute(dataset, entry.getKey(), entry.getValue());
	}

	private static <K, V> Map<K, V> with(Map<K, V> map, K key, V value) {
		map.put(key, value);
		return map;
	}

	private static class N5WriterSupplier implements Supplier<N5Writer>, Serializable {

		private final String container;

		private N5WriterSupplier(String container) {
			this.container = container;
		}

		@Override
		public N5Writer get() {
			try {
				return Files.isDirectory(Paths.get(container))
						? new N5FSWriter(container)
						: new N5HDF5Writer(container);
			} catch (final IOException e) {
				throw new RuntimeException(e);
			}
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

}
