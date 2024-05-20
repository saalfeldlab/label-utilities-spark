package org.janelia.saalfeldlab.label.spark.affinities;

import com.google.gson.annotations.Expose;
import net.imglib2.FinalInterval;
import net.imglib2.FinalRealInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.util.ConstantUtils;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.function.BiFunction;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

public class MakePredictionMask {

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	public static final String NETWORK_SIZE_DIFF_KEY = "networkSizeDiff";

	private static class Args implements Callable<Void> {

		@Expose
		@CommandLine.Option(names = "--mask-container", paramLabel = "MASK_CONTAINER", description = "Path to N5 container with affinities dataset.", required = true)
		String maskContainer = null;

		@Expose
		@CommandLine.Option(names = "--input-container", paramLabel = "INPUT_CONTAINER", description = "Path to N5 container with input dataset. Defaults to MASK_CONTAINER", required = false)
		String inputContainer = null;

		@Expose
		@CommandLine.Option(names = "--input-dataset", paramLabel = "INPUT_DATASET", description = "Input dataset either mask or raw data. If raw data, only dimensions are relevant.")
		String inputDataset = null;

		@Expose
		@CommandLine.Option(names = "--input-dataset-size", paramLabel = "INPUT_DATASET_SIZE", description = "In voxels. One of INPUT_DATASET_SIZE and INPUT_DATASET must be specified", split = ",")
		long[] inputDatasetSize;

		@Expose
		@CommandLine.Option(names = "--input-is-mask", defaultValue = "false")
		Boolean inputIsMask;

		@Expose
		@CommandLine.Option(names = "--mask-dataset", paramLabel = "MASK_DATASET", description = "Path to mask dataset in mask container. Will be written into", required = true)
		String maskDataset = null;

		@Expose
		@CommandLine.Option(names = "--output-dataset-size", paramLabel = "OUTPUT_DATASET_SIZE", description = "In voxels. ", required = true, split = ",")
		long[] outputDatasetSIze;

		@Expose
		@CommandLine.Option(names = "--input-resolution", paramLabel = "INPUT_RESOLUTION", defaultValue = "36,36,360", split = ",")
		double[] inputResolution;

		@Expose
		@CommandLine.Option(names = "--output-resolution", paramLabel = "OUTPUT_RESOLUTION", defaultValue = "108,108,120", split = ",")
		double[] outputResolution;

		@Expose
		@CommandLine.Option(names = "--input-offset", paramLabel = "INPUT_OFFSET", defaultValue = "0,0,0", split = ",")
		double[] inputOffset;

		@Expose
		@CommandLine.Option(names = "--output-offset", paramLabel = "OUTPUT_OFFSET", defaultValue = "0,0,0", split = ",")
		double[] outputOffset;

		@Expose
		@CommandLine.Option(names = "--network-input-size", paramLabel = "INPUT_SIZE", split = ",", description = "World coordinates", defaultValue = "15480,15480,15480")//"31032,31032,32760")
		double[] networkInputSizeWorld;

		@Expose
		@CommandLine.Option(names = "--network-output-size", paramLabel = "OUTPUT_SIZE", split = ",", description = "World coordinates", defaultValue = "7800,7560,7560")//"23112, 23112, 24840")
		double[] networkOutputSizeWorld;

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
		public Void call() throws Exception {

			//			rawContainer = rawContainer == null ? maskContainer : rawContainer;
			//
			//			predictionContainer = predictionContainer == null ? maskContainer : predictionContainer;
			if (inputDatasetSize == null && inputDataset == null)
				throw new Exception("One of input dataset size or input dataset must be specified!");

			if (inputDataset != null)
				inputDatasetSize = new N5FSReader(inputContainer).getDatasetAttributes(inputDataset).getDimensions();

			for (int d = 0; d < inputOffset.length; ++d)
				if (inputOffset[d] / inputResolution[d] != (int)(inputOffset[d] / inputResolution[d]))
					throw new Exception("Offset not integer multiple of resolution!");

			for (int d = 0; d < outputOffset.length; ++d)
				if (outputOffset[d] / outputResolution[d] != (int)(outputOffset[d] / outputResolution[d]))
					throw new Exception("Offset not integer multiple of resolution!");

			return null;
		}

		public int[] blockSize() {

			return asInt(divide(networkOutputSizeWorld, outputResolution));

		}

		public Supplier<RandomAccessible<UnsignedByteType>> inputMaskSupplier() {

			if (inputDataset == null || !inputIsMask) {
				return new MaskProviderFromDims(inputDatasetSize);
			} else {
				return new MaskProviderFromN5(
						new N5WriterSupplier(inputContainer, false, false),
						inputDataset);
			}
		}

		public double[] snapDimensionsToBlockSize(final double[] dimensionsWorld, final double[] blockSizeWorld) {

			final double[] snapped = new double[dimensionsWorld.length];
			Arrays.setAll(snapped, d -> Math.ceil(dimensionsWorld[d] / blockSizeWorld[d]) * blockSizeWorld[d]);
			return snapped;
		}

		public double[] inputSizeWorld() {

			final double[] inputSizeWorld = new double[inputDatasetSize.length];
			Arrays.setAll(inputSizeWorld, d -> inputDatasetSize[d] * inputResolution[d]);
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

		run(argv);
	}

	public static void run(String[] argv) throws IOException {

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
		final long[] outputDatasetSize = args.outputDatasetSIze;

		final long[] validInputSizeInOutputCoordinates = convertAsLong(divide(inputSizeWorld, args.outputResolution), Math::floor);

		final N5WriterSupplier n5out = new N5WriterSupplier(args.maskContainer, true, true);
		n5out.get().createDataset(args.maskDataset, outputDatasetSize, args.blockSize(), DataType.UINT8, new GzipCompression());
		n5out.get().setAttribute(args.maskDataset, NETWORK_SIZE_DIFF_KEY, networkSizeDiff);
		n5out.get().setAttribute(args.maskDataset, "resolution", args.outputResolution);
		n5out.get().setAttribute(args.maskDataset, "offset", args.outputOffset);
		n5out.get().setAttribute(args.maskDataset, "min", 0);
		n5out.get().setAttribute(args.maskDataset, "max", 1);
		n5out.get().setAttribute(args.maskDataset, "value_range", new double[]{0, 1});

		run(
				n5out,
				args.maskDataset,
				args.inputResolution,
				args.outputResolution,
				args.inputOffset,
				args.outputOffset,
				networkSizeDiffHalfWorld,
				args.inputMaskSupplier(),
				outputDatasetSize,
				args.blockSize());

	}

	private static void run(
			final Supplier<? extends N5Writer> n5out,
			final String maskDataset,
			final double[] inputVoxelSize,
			final double[] outputVoxelSize,
			final double[] inputOffset,
			final double[] outputOffset,
			final double[] paddingInWorldCoordinates,
			final Supplier<RandomAccessible<UnsignedByteType>> inputMask,
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
						final long[] min = block._1()._1();
						final long[] max = block._1()._2();
						final Interval interval = new FinalInterval(min, max);
						final RandomAccessible<UnsignedByteType> mask = inputMask.get();
						final DatasetAttributes attributes = new DatasetAttributes(outputDatasetSize, blockSize, DataType.UINT8, new GzipCompression());
						final double[] minReal = LongStream.of(min).asDoubleStream().toArray();
						final double[] maxReal = LongStream.of(max).asDoubleStream().toArray();
						//						final Scale outputScale = new Scale(outputVoxelSize);
						//						final Scale inputScale = new Scale(inputVoxelSize);
						final AffineTransform3D outputTransform = new AffineTransform3D();
						outputTransform.set(outputVoxelSize[0], 0, 0);
						outputTransform.set(outputVoxelSize[1], 1, 1);
						outputTransform.set(outputVoxelSize[2], 2, 2);
						outputTransform.setTranslation(outputOffset);
						final AffineTransform3D inputTransform = new AffineTransform3D();
						inputTransform.set(inputVoxelSize[0], 0, 0);
						inputTransform.set(inputVoxelSize[1], 1, 1);
						inputTransform.set(inputVoxelSize[2], 2, 2);
						inputTransform.setTranslation(inputOffset);
						outputTransform.apply(minReal, minReal);
						outputTransform.apply(maxReal, maxReal);
						Arrays.setAll(minReal, d -> minReal[d] - paddingInWorldCoordinates[d]);
						Arrays.setAll(maxReal, d -> maxReal[d] + paddingInWorldCoordinates[d]);
						inputTransform.applyInverse(minReal, minReal);
						inputTransform.applyInverse(maxReal, maxReal);
						boolean isForeground = true;
						final IntervalView<UnsignedByteType> inputInterval = Views.interval(mask, Intervals.smallestContainingInterval(new FinalRealInterval(minReal, maxReal)));
						LOG.debug("Checking interval ({} {}) for block ({} {})", Intervals.minAsLongArray(inputInterval), Intervals.maxAsLongArray(inputInterval), min, max);
						// TODO use integral images instead of discarding entire blocks
						for (final UnsignedByteType m : inputInterval) {
							if (m.get() == 0) {
								isForeground = false;
								break;
							}
						}

						final ArrayImg<UnsignedByteType, ByteArray> outputMask = ArrayImgs.unsignedBytes(Intervals.dimensionsAsLongArray(interval));
						Arrays.fill(outputMask.update(null).getCurrentStorageArray(), isForeground ? (byte)1 : 0);
						// this is bad alignment area in cremi sample_A+
						//						for (long z = Math.max(min[2], 344); z < Math.min(max[2], 357); ++z)
						//							Views.hyperSlice(Views.translate(outputMask, min), 2, z).forEach(UnsignedByteType::setZero);

						N5Utils.saveBlock(
								outputMask,
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
		Arrays.setAll(ceil, d -> (long)converter.applyAsDouble(array[d]));
		return ceil;
	}

	private static int[] convertAsInt(double[] array, DoubleUnaryOperator converter) {

		final int[] ceil = new int[array.length];
		Arrays.setAll(ceil, d -> (int)converter.applyAsDouble(array[d]));
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

	private static class MaskProviderFromDims implements Supplier<RandomAccessible<UnsignedByteType>>, Serializable {

		private final long[] dims;

		private MaskProviderFromDims(long[] dims) {

			this.dims = dims;
		}

		@Override
		public RandomAccessible<UnsignedByteType> get() {

			return Views.extendZero(ConstantUtils.constantRandomAccessibleInterval(new UnsignedByteType(1), dims.length, new FinalInterval(dims)));
		}
	}

	private static class MaskProviderFromN5 implements Supplier<RandomAccessible<UnsignedByteType>>, Serializable {

		private final N5WriterSupplier n5;

		private final String dataset;

		private MaskProviderFromN5(N5WriterSupplier n5, String dataset) {

			this.n5 = n5;
			this.dataset = dataset;
		}

		@Override
		public RandomAccessible<UnsignedByteType> get() {

			final RandomAccessibleInterval<IntegerType<?>> img = (RandomAccessibleInterval)N5Utils.open(n5.get(), dataset);
			final RandomAccessibleInterval<UnsignedByteType> convertedImg = Converters.convert(img, (s, t) -> t.setInteger(s.getIntegerLong()), new UnsignedByteType());
			return Views.extendValue(convertedImg, new UnsignedByteType(0));
		}

	}

}
