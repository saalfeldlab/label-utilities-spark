package org.janelia.saalfeldlab.label.spark.affinities;

import com.google.gson.annotations.Expose;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.util.ConstantUtils;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import scala.Tuple2;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.function.BiFunction;
import java.util.function.DoubleToLongFunction;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class MakePredictionMask {

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	private static class Args implements Callable<Void> {

		@Expose
		@CommandLine.Option(names = "--mask-container", paramLabel = "MASK_CONTAINER", description = "Path to N5 container with affinities dataset.", required = true)
		String maskContainer = null;

		@Expose
		@CommandLine.Option(names = "--mask-dataset", paramLabel = "MASK_DATASET", description = "Path to mask dataset in mask container. Will be written into", required=true)
		String maskDataset = null;

		@Expose
		@CommandLine.Option(names = "--input-size-voxels", paramLabel = "INPUT_SIZE_VOXELS", split =",", required=true)
		long[] inputSizeVoxels;

		@Expose
		@CommandLine.Option(names = "--input-resolution", paramLabel = "INPUT_RESOLUTION", defaultValue = "36,36,360", split = ",")
		double[] inputResolution;

		@Expose
		@CommandLine.Option(names = "--output-resolution", paramLabel = "OUTPUT_RESOLUTION", defaultValue = "108,108,120", split = ",")
		double[] outputResolution;

		@Expose
		@CommandLine.Option(names = "--network-input-size", paramLabel = "INPUT_SIZE", split = ",", description = "World coordinates", defaultValue = "31032,31032,32760")
		double[] networkInputSizeWorld;

		@Expose
		@CommandLine.Option(names = "--network-output-size", paramLabel = "OUTPUT_SIZE", split = ",", description = "World coordinates", defaultValue = "23112, 23112, 24840")
		double[] networkOutputSizeWorld;

		@Expose
		@CommandLine.Option(names = "--block-size", paramLabel = "BLOCK_SIZE", description = "defaults to OUTPUT_SIZE / OUTPUT_RESOLUTION")
		int[] blockSize = null;

		@Expose
		@CommandLine.Option(names = "--blocks-per-task", defaultValue = "1,1,1", split = ",")
		int[] blocksPerTask;

//		@Expose
//		@CommandLine.Option(names = "--raw-container", paramLabel = "RAW_CONTAINER", description = "Path to raw container. Defaults to MASK_CONTAINER.")
//		String rawContainer = null;
//
//		@Expose
//		@CommandLine.Option(names = "prediction-container", paramLabel = "PREDICTION_CONTAINER", description = "Path to prediction container. Defaults to MASK_CONTAINER")
//		String predictionContainer = null;
//
//		@Expose
//		@CommandLine.Option(names = "--raw-dataset", paramLabel = "RAW_DATASET", description = "Path of raw dataset dataset in RAW_CONTAINER.", defaultValue = "volumes/raw")
//		String rawDataset;
//
//		@Expose
//		@CommandLine.Option(names = "--prediction-dataset", paramLabel = "PREDICTION_DATASET", description = "Path to prediction dataset in PREDICTION_CONTAINER", defaultValue = "volumes/affinities/predictions")
//		String predictionDataset;


		@Override
		public Void call() {

//			rawContainer = rawContainer == null ? maskContainer : rawContainer;
//
//			predictionContainer = predictionContainer == null ? maskContainer : predictionContainer;

			blockSize = blockSize == null ? asInt(divide(networkOutputSizeWorld, outputResolution)) : blockSize;

			return null;
		}

		public double[] snapDimensionsToBlockSize(final double[] dimensionsWorld, final double[] blockSizeWorld) {
			final double[] snapped = new double[dimensionsWorld.length];
			Arrays.setAll(snapped, d -> Math.ceil(dimensionsWorld[d] / blockSizeWorld[d]) * blockSizeWorld[d]);
			return snapped;
		}

		public double[] inputSizeWorld() {
			final double[] inputSizeWorld = new double[inputSizeVoxels.length];
			Arrays.setAll(inputSizeWorld, d -> inputSizeVoxels[d] * inputResolution[d]);
			return inputSizeWorld;
		}

		public double[] inputSizeWorldSnapped() {
			return snapDimensionsToBlockSize(inputSizeWorld(), networkOutputSizeWorld);
		}

		public double[] networkSizeDiffWorld() {
			final double[] diff = new double[this.networkInputSizeWorld.length];
			Arrays.setAll(diff, d -> networkInputSizeWorld[d] - networkOutputSizeWorld[d]);
			return diff;
		}

		public double[] networkSizeDiffHalfWorld() {
			final double[] diff = networkSizeDiffWorld();
			Arrays.setAll(diff, d -> diff[d] / 2);
			return diff;
		}

		public double[] networkOutputSize() {
			return divide(this.networkInputSizeWorld, outputResolution);
		}
	}

	public static void main(String[] argv) throws IOException {

		final Args args = new Args();
		CommandLine.call(args, argv);

		final double[] inputSizeWorld = args.inputSizeWorld();
		final double[] inputSizeWorldSnappedToOutput = args.inputSizeWorldSnapped();
		final double[] networkSizeDiffWorld = args.networkSizeDiffWorld();
		final double[] networkSizeDiffHalfWorld = args.networkSizeDiffHalfWorld();
		final long[] networkSizeDiff = asLong(divide(networkSizeDiffWorld, args.outputResolution));

//		final double[] validMin = networkSizeDiffHalfWorld.clone();
//		final double[] validMax = subtract(inputSizeWorld, networkSizeDiffHalfWorld);

		final double[] outputDatasetSizeDouble = divide(inputSizeWorldSnappedToOutput, args.outputResolution);
		final long[] outputDatasetSize = asLong(outputDatasetSizeDouble);

		final long[] validInputSizeInOutputCoordinates = convertAsLong(divide(inputSizeWorld, args.outputResolution), Math::floor);

		final N5WriterSupplier n5out = new N5WriterSupplier(args.maskContainer, true, true);
		n5out.get().createDataset(args.maskDataset, outputDatasetSize, args.blockSize, DataType.UINT8, new GzipCompression());
		n5out.get().setAttribute(args.maskDataset, "networkSizeDiff", networkSizeDiff);
		n5out.get().setAttribute(args.maskDataset, "resolution", args.outputResolution);

		run(n5out, args.maskDataset, validInputSizeInOutputCoordinates, outputDatasetSize, args.blockSize);

	}

	private static void run(
			final Supplier<? extends N5Writer> n5out,
			final String maskDataset,
			final long[] validDimensions,
			final long[] outputDatasetSize,
			final int[] blockSize) {
		final SparkConf conf = new SparkConf().setAppName(MethodHandles.lookup().lookupClass().getName());
		try (final JavaSparkContext sc = new JavaSparkContext(conf)) {
			final List<Tuple2<Tuple2<long[], long[]>, long[]>> blocks = Grids
					.collectAllContainedIntervalsWithGridPositions(outputDatasetSize, blockSize)
					.stream()
					.map(i -> new Tuple2<>(toMinMaxTuple(i.getA(), Tuple2::new), i.getB()))
					.collect(Collectors.toList());
			sc
					.parallelize(blocks)
					.foreach(block -> {
						final Interval interval = new FinalInterval(block._1()._1(), block._1()._2());
						final RandomAccessibleInterval<UnsignedByteType> insideMask = ConstantUtils.constantRandomAccessibleInterval(
								new UnsignedByteType(1),
								interval.numDimensions(),
								new FinalInterval(validDimensions));
						LOG.debug("Total dimensions {} valid Dimensions {} interval ({} {})", outputDatasetSize, validDimensions, Intervals.minAsLongArray(interval), Intervals.maxAsLongArray(interval));
						final DatasetAttributes attributes = new DatasetAttributes(outputDatasetSize, blockSize, DataType.UINT8, new GzipCompression());
						N5Utils.saveBlock(
								Views.zeroMin(Views.interval(Views.extendZero(insideMask), interval)),
								n5out.get(),
								maskDataset,
								attributes,
								block._2());
					});
		}


	}

	private static long[] asLong(double[] array) {
		return convertAsLong(array, d -> d);
	}

	private static int[] asInt(double[] array) {
		return convertAsInt(array, d -> d);
	}

	private static long[] convertAsLong(double[] array, DoubleUnaryOperator converter) {
		final long[] ceil = new long[array.length];
		Arrays.setAll(ceil, d -> (long) converter.applyAsDouble(array[d]));
		return ceil;
	}

	private static int[] convertAsInt(double[] array, DoubleUnaryOperator converter) {
		final int[] ceil = new int[array.length];
		Arrays.setAll(ceil, d -> (int) converter.applyAsDouble(array[d]));
		return ceil;
	}


	private static double[] convert(double[] array, DoubleUnaryOperator converter) {
		final double[] ceil = new double[array.length];
		Arrays.setAll(ceil, d -> converter.applyAsDouble(array[d]));
		return ceil;
	}

	private static double[] subtract(final double[] minuend, final double[] subtrahend) {
		final double[] difference = new double[minuend.length];
		Arrays.setAll(difference, d -> minuend[d] - subtrahend[d]);
		return difference;
	}

	private static double[] divide(final double[] a, final double[] b) {
		final double[] quotient = new double[a.length];
		Arrays.setAll(quotient, d -> a[d] / b[d]);
		return quotient;
	}

	private static <T> T toMinMaxTuple(final Interval interval, BiFunction<long[], long[], T> toTuple) {
		return toTuple.apply(Intervals.minAsLongArray(interval), Intervals.maxAsLongArray(interval));
	}



}
