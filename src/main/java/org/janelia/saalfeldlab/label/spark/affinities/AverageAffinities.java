package org.janelia.saalfeldlab.label.spark.affinities;

import com.google.gson.annotations.Expose;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealDoubleConverter;
import net.imglib2.converter.RealFloatConverter;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import pl.touk.throwing.ThrowingBiConsumer;
import scala.Tuple2;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class AverageAffinities {

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	private static class Args implements Callable<Void> {

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
		@CommandLine.Option(names = "--averaged-affinity-dataset", paramLabel = "AFFINITIES", description = "Path of affinities dataset in INPUT_CONTAINER.")
		String averaged = null;

		@Expose
		@CommandLine.Option(names = "--offsets", arity = "1..*", paramLabel = "OFFSETS", description = "Structuring elements for affinities. Defaults to -1,0,0 0,-1,0 0,0,-1.", converter = Offset.Converter.class)
		Offset[] offsets = {new Offset(-1, 0, 0), new Offset(0, -1, 0), new Offset(0, 0, -1)};

		@Expose
		@CommandLine.Option(names = "--block-size", paramLabel = "BLOCK_SIZE", description = "Block size of output. Defaults to input block size w/o channel dim if not specified.", split = ",")
		int[] blockSize = null;

		@Expose
		@CommandLine.Option(names = "--blocks-per-task", paramLabel = "BLOCKS_PER_TASK", description = "How many blocks to combine for watersheds/connected components (one value per dimension)", split=",")
		int[] blocksPerTask = {1, 1, 1};

		@CommandLine.Option(names = "--json-pretty-print", defaultValue = "true")
		transient Boolean prettyPrint;

		@CommandLine.Option(names = "--json-disable-html-escape", defaultValue = "true")
		transient Boolean disbaleHtmlEscape;

		@Override
		public Void call() throws Exception {

			final long numSpecifiedOffsetChannelIndices = Stream.of(offsets).filter(o -> o.channelIndex() >= 0).count();
			final DatasetAttributes attributes = new N5FSWriter(inputContainer).getDatasetAttributes(affinities);

			if (numSpecifiedOffsetChannelIndices == 0) {
				if (attributes.getDimensions()[attributes.getNumDimensions() - 1] != this.offsets.length)
					throw new Exception("Need to define all offsets when not specifying channel indices explicitly.");
			} else if (numSpecifiedOffsetChannelIndices < offsets.length - 1)
				throw new Exception("Can only specify all or no channel indices for offsets.");

			if (inputContainer.equalsIgnoreCase(outputContainer) && affinities.equalsIgnoreCase(averaged))
				throw new Exception("Input and output are the same!");

			if (averaged == null)
				averaged = affinities + "-averaged";

			if (outputContainer == null)
				outputContainer = inputContainer;

			if (blockSize == null)
				blockSize = subArray(attributes.getBlockSize(), 0, attributes.getNumDimensions() - 1);

			return null;
		}

		public Offset[] enumeratedOffsets() {
			final Offset[] enumeratedOffsets = new Offset[this.offsets.length];
			for (int i = 0; i < offsets.length; ++i) {
				final Offset o = this.offsets[i];
				enumeratedOffsets[i] = new Offset(
						o.channelIndex() == -1 ? i : o.channelIndex(),
						o.offset());
			}
			return enumeratedOffsets;
		}
	}

	public static void main(String[] argv) throws IOException {

		final Args args = new Args();
		CommandLine.call(args, argv);



		final N5WriterSupplier n5InSupplier = new N5WriterSupplier(args.inputContainer, false, false);
		final DatasetAttributes inputAttributes = n5InSupplier.get().getDatasetAttributes(args.affinities);
		final N5WriterSupplier n5OutSupplier = new N5WriterSupplier(args.outputContainer, args.prettyPrint, args.disbaleHtmlEscape);

		n5OutSupplier.get().createDataset(args.averaged, ignoreLast(inputAttributes.getDimensions()), args.blockSize, DataType.FLOAT32, new GzipCompression());
		n5InSupplier.get().listAttributes(args.affinities).forEach(ThrowingBiConsumer.unchecked((key, clazz) -> n5OutSupplier.get().setAttribute(args.averaged, key, n5InSupplier.get().getAttribute(args.affinities, key, clazz))));
		n5OutSupplier.get().createDataset(args.averaged, ignoreLast(inputAttributes.getDimensions()), args.blockSize, DataType.FLOAT32, new GzipCompression());

		final SparkConf conf = new SparkConf().setAppName(MethodHandles.lookup().lookupClass().getSimpleName());
		try (final JavaSparkContext sc = new JavaSparkContext(conf)) {
			run(
					sc,
					args.blockSize,
					args.blocksPerTask,
					n5InSupplier,
					n5OutSupplier,
					args.affinities,
					args.averaged,
					args.enumeratedOffsets());
		}

	}

	private static void run(
			final JavaSparkContext sc,
			final int[] blockSize,
			final int[] blocksPerTask,
			final N5WriterSupplier n5InSupplier,
			final N5WriterSupplier n5OutSupplier,
			final String affinities,
			final String averaged,
			final Offset[] enumeratedOffsets) throws IOException {

		final int[] taskSize = IntStream.range(0, blockSize.length).map(i -> blockSize[i] * blocksPerTask[i]).toArray();

		final List<Tuple2<long[], long[]>> blocks = Grids
				.collectAllContainedIntervals(n5OutSupplier.get().getDatasetAttributes(averaged).getDimensions(), taskSize)
				.stream()
				.map(i -> new Tuple2<>(Intervals.minAsLongArray(i), Intervals.maxAsLongArray(i)))
				.collect(Collectors.toList());

		LOG.info("Parallelizing over blocks {}", blocks);


		sc
				.parallelize(blocks)
				.mapToPair(block -> new Tuple2<>(block, (RandomAccessibleInterval<FloatType>) N5Utils.open(n5OutSupplier.get(), affinities)))
				.map(p -> {
					final long[] min = p._1()._1();
					final long[] max = p._1()._2();
					final RandomAccessibleInterval<DoubleType> averagedAffinities = ArrayImgs.doubles(Intervals.dimensionsAsLongArray(new FinalInterval(min, max)));
					final RandomAccessibleInterval<DoubleType> slice1 = Views.translate(averagedAffinities, min);
					for (final Offset offset : enumeratedOffsets) {
						final RandomAccessible<FloatType> affs = Views.extendZero(Views.hyperSlice(p._2(), min.length, (long) offset.channelIndex()));
						final IntervalView<DoubleType> expanded1 = Views.interval(Views.extendZero(slice1), expandAsNeeded(slice1, offset.offset()));
						final IntervalView<DoubleType> expanded2 = Views.translate(expanded1, offset.offset());

						LOG.info(
								"Averaging {} voxels for offset {} : [{}:{}] ({})",
								Intervals.numElements(expanded1),
								offset,
								Intervals.minAsLongArray(expanded1),
								Intervals.minAsLongArray(expanded1),
								Intervals.dimensionsAsLongArray(expanded1));

						LoopBuilder
								.setImages(Views.interval(Converters.convert(affs, new RealDoubleConverter<>(), new DoubleType()), expanded1), expanded1, expanded2)
								.forEachPixel((a, s1, s2) -> {
									if (Double.isFinite(a.getRealDouble())) {
										s1.add(a);
										s2.add(a);
									}
								});
					}

					final double factor = 0.5 / enumeratedOffsets.length;
					Views.iterable(slice1).forEach(px -> px.mul(factor));

					final N5Writer n5 = n5OutSupplier.get();
					final DatasetAttributes attributes = n5.getDatasetAttributes(averaged);
					final long[] gridOffset = min.clone();
					Arrays.setAll(gridOffset, d -> gridOffset[d] / attributes.getBlockSize()[d]);

					N5Utils.saveBlock(
							Converters.convert(slice1, new RealFloatConverter<>(), new FloatType()),
							n5OutSupplier.get(),
							averaged,
							attributes,
							gridOffset);

					return true;
				})
				.count();

	}
	
	private static long[] ignoreLast(final long[] dims) {
		final long[] newDims = new long[dims.length - 1];
		Arrays.setAll(newDims, d -> dims[d]);
		return newDims;
	}

	private static Interval translate(Interval interval, final long[] translation) {
		for (int d = 0; d < translation.length; ++d)
			interval = Intervals.translate(interval, translation[d], d);
		return interval;
	}

	private static long[] abs(long... array) {
		final long[] abs = new long[array.length];
		Arrays.setAll(abs, d -> Math.abs(array[d]));
		return abs;
	}

	private static Interval expandAsNeeded(
			final Interval source,
			final long[] offsets
	) {
		final long[] min = Intervals.minAsLongArray(source);
		final long[] max = Intervals.maxAsLongArray(source);

		for (int d = 0; d < offsets.length; ++d) {
			final long offset = offsets[d];
			if (offset < 0)
				max[d] -= offset;
			else if (offset > 0)
				min[d] -= offset;
		}

		return new FinalInterval(min, max);

	}

	private static int[] subArray(final int[] array, int start, int stop) {
		final int[] result = new int[stop - start];
		Arrays.setAll(result, d -> array[d] + start);
		return result;
	}

}
