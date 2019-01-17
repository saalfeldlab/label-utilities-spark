package org.janelia.saalfeldlab.label.spark.affinities;

import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.labeling.affinities.ConnectedComponents;
import net.imglib2.algorithm.labeling.affinities.Watersheds;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.converter.Converters;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.label.Label;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.logic.BoolType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.ConstantUtils;
import net.imglib2.util.IntervalIndexer;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import net.imglib2.view.composite.CompositeIntervalView;
import net.imglib2.view.composite.RealComposite;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import scala.Tuple2;
import scala.Tuple3;

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
import java.util.stream.LongStream;
import java.util.stream.Stream;

public class SparkWatersheds {

	private static final String RESOLUTION_KEY = "resolution";

	private static final String OFFSET_KEY = "offset";

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	private static class Offset {

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

		@CommandLine.Parameters(arity = "1", paramLabel = "INPUT_CONTAINER", description = "Path to N5 container with affinities dataset.")
		String inputContainer = null;

		@CommandLine.Option(names = "--output-container", paramLabel = "OUTPUT_CONTAINER", description = "Path to output container. Defaults to INPUT_CONTAINER.")
		String outputContainer = null;

		@CommandLine.Option(names = "--affinity-dataset", paramLabel = "AFFINITIES", description = "Path of affinities dataset in INPUT_CONTAINER.")
		String affinities = "volumes/affinities/prediction";

		@CommandLine.Option(names = "--connected-components-dataset", paramLabel = "CONNECTED_COMPONENTS", description = "Path to connected components in OUTPUT_CONTAINER")
		String connectedComponents = "volumes/labels/connected_components";

		@CommandLine.Option(names = "--watersheds-dataset", paramLabel = "WATERSHEDS", description = "Path to watersheds in OUTPUT_CONTAINER")
		String watersheds = "volumes/labels/watersheds";

		@CommandLine.Option(names = "--watershed-seeds-mask-dataset", paramLabel = "WATERSHED_SEEDS_MASK", description = "Path to watershed seeds mask in OUTPUT_CONTAINER")
		String watershedSeedsMask = "volumes/labels/watershed_seeds";

		@CommandLine.Option(names = "--invert-affinities-axis", paramLabel = "INVERT_AFFINITIES_AXIS", description = "Invert axis that holds affinities. This is necessary if affinities were generated as [z,y,x].")
		Boolean invertAffinitiesAxis = false;

		@CommandLine.Option(names = "--threshold", paramLabel = "THRESHOLD", description = "Threshold for thresholding affinities. Defaults to 0.5.")
		Double threshold = 0.5;

		@CommandLine.Option(names = "--offsets", arity = "1..*", paramLabel = "OFFSETS", description = "Structuring elements for affinities. Defaults to -1,0,0 0,-1,0 0,0,-1.")
		Offset[] offsets = {new Offset(-1, 0, 0), new Offset(0, -1, 0), new Offset(0, 0, -1)};

		@CommandLine.Option(names = "--block-size", paramLabel = "BLOCK_SIZE", description = "Block size of output.", split = ",")
		int[] blockSize = {64, 64, 64};

		@CommandLine.Option(names = "--blocks-per-task", paramLabel = "BLOCKS_PER_TASK", description = "How many blocks to combine for watersheds/connected components (one value per dimension)", split=",")
		int[] blocksPerTask = {1, 1, 1};

		@CommandLine.Option(names = "--halo", paramLabel = "HALO", description = "Include halo region to run connected components/watersheds", split=",")
		int[] halo = {0, 0, 0};

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

		prepareOutputDatasets(
				n5out.get(),
				outputDims,
				args.blockSize,
				args.connectedComponents,
				args.watershedSeedsMask,
				args.watersheds,
				Optional.ofNullable(n5in.get().getAttribute(args.affinities, RESOLUTION_KEY, double[].class)).orElse(ones(outputDims.length)),
				Optional.ofNullable(n5in.get().getAttribute(args.affinities, OFFSET_KEY, double[].class)).orElse(new double[outputDims.length]),
				new HashMap<>());


		final SparkConf conf = new SparkConf().setAppName(MethodHandles.lookup().lookupClass().getName());
		try (final JavaSparkContext sc = new JavaSparkContext(conf)) {
			run(
					sc,
					n5in,
					n5out,
					outputDims,
					IntStream.of(args.halo).mapToLong(i -> i).toArray(),
					args.invertAffinitiesAxis,
					args.threshold,
					Stream.of(args.offsets).map(Offset::offset).toArray(long[][]::new),
					args.affinities,
					args.connectedComponents,
					args.watershedSeedsMask,
					args.watersheds,
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
			final double threshold,
			final long[][] offsets,
			final String affinities,
			final String connectedComponents,
			final String watershedSeedsMask,
			final String watersheds,
			final int[] blockSize,
			final int[] blocksPerTask) throws IOException {

		final int numChannels = offsets.length;
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
						final IntervalView<FloatType> translatedSlice = Views.translate(slice, offsets[index]);
						final Cursor<FloatType> c = Views.iterable(translatedSlice).cursor();
						while (c.hasNext()) {
							c.fwd();
							if (!Intervals.contains(slice, c))
								c.get().setReal(Double.NaN);
						}
					}
					return affs;
				})
				.mapToPair(t -> {
					final RandomAccessibleInterval<FloatType> uncollapsedAffinities = t._2();
					final CompositeIntervalView<FloatType, RealComposite<FloatType>> affs = Views.collapseReal(uncollapsedAffinities);
					final RandomAccessibleInterval<BoolType> mask = ConstantUtils.constantRandomAccessibleInterval(new BoolType(true), affs.numDimensions(), affs);
					final ArrayImg<UnsignedLongType, LongArray> labels = ArrayImgs.unsignedLongs(Intervals.dimensionsAsLongArray(affs));
					labels.forEach(it -> it.set(Label.INVALID));
					final ArrayImg<BitType, LongArray> unionFindMask = ArrayImgs.bits(Intervals.dimensionsAsLongArray(affs));
					LOG.info("Connected components on interval {}/{} with threshold {}", toString(labels), toString(affs), threshold);
					final CellGrid grid = new CellGrid(outputDims, blockSize);
					final long[] cellPos = Intervals.minAsLongArray(t._1());
					grid.getCellPosition(cellPos, cellPos);
					final long minId = IntervalIndexer.positionToIndex(cellPos, grid.getGridDimensions()) * Intervals.numElements(blockSize) + 1;
					final long maxId = ConnectedComponents.fromSymmetricAffinities(
							Views.extendValue(mask, new BoolType(false)),
							affs,
							labels,
							Views.extendZero(unionFindMask),
							threshold,
							offsets,
							id -> id + minId);
					final DatasetAttributes labelsAttrs = new DatasetAttributes(outputDims, blockSize, DataType.UINT64, new GzipCompression());
					LOG.info("Saving interval {} at min {} and max id {}", toString(Intervals.expand(labels, negativeHalo)), Intervals.minAsLongArray(t._1()), maxId);
					long[] blockOffset = Intervals.minAsLongArray(t._1());
					grid.getCellPosition(blockOffset, blockOffset);

					N5Utils.saveBlock(Views.interval(labels, Intervals.expand(labels, negativeHalo)), n5out.get(), connectedComponents, labelsAttrs, blockOffset);

					final RandomAccessibleInterval<BitType> watershedSeedsMaskImg = ArrayImgs.bits(Intervals.dimensionsAsLongArray(unionFindMask));
					Watersheds.seedsFromMask(Views.extendValue(unionFindMask, new BitType(true)), watershedSeedsMaskImg, Watersheds.symmetricOffsets(offsets));
					final List<Point> seeds = Watersheds.collectSeeds(watershedSeedsMaskImg);

					final RandomAccessibleInterval<ByteType> watershedSeedsMaskAsByte = Converters.convert(watershedSeedsMaskImg, (src, tgt) -> tgt.setInteger(src.get() ? 1 : 0), new ByteType());

					final DatasetAttributes watershedSeedsMaskAttrs = new DatasetAttributes(outputDims, blockSize, DataType.INT8, new GzipCompression());
					N5Utils.saveBlock(Views.interval(watershedSeedsMaskAsByte, Intervals.expand(watershedSeedsMaskAsByte, negativeHalo)), n5out.get(), watershedSeedsMask, watershedSeedsMaskAttrs, blockOffset);

					final ImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
					final RandomAccessibleInterval<FloatType> symmetricAffinities = Watersheds.constructAffinities(uncollapsedAffinities, offsets, factory);
					// TODO use different priority queue to make more efficient
					Watersheds.seededFromAffinities(Views.collapseReal(symmetricAffinities), labels, seeds, offsets);

					final DatasetAttributes watershedsAttrs = new DatasetAttributes(outputDims, blockSize, DataType.UINT64, new GzipCompression());
					N5Utils.saveBlock(Views.interval(labels, Intervals.expand(labels, negativeHalo)), n5out.get(), watersheds, watershedsAttrs, blockOffset);

					return new Tuple2<>(t._1(), new Tuple3<>(uncollapsedAffinities, unionFindMask, labels));
				})
				.count();
		;

	}

	private static void prepareOutputDatasets(
			final N5Writer n5,
			final long[] dims,
			final int[] blockSize,
			final String connectedComponents,
			final String watershedSeedsMask,
			final String watersheds,
			final double[] resolution,
			final double[] offset,
			final Map<String, Object> additionalData
	) throws IOException {

		additionalData.put(RESOLUTION_KEY, resolution);
		additionalData.put(OFFSET_KEY, offset);
		prepareOutputDataset(n5, connectedComponents, dims, blockSize, DataType.UINT64, additionalData);
		prepareOutputDataset(n5, watershedSeedsMask, dims, blockSize, DataType.INT8, with(with(additionalData, "min", 0.0), "max", 1.0));
		prepareOutputDataset(n5, watersheds, dims, blockSize, DataType.UINT64, additionalData);
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
