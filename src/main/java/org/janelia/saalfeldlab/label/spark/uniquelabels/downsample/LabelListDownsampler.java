package org.janelia.saalfeldlab.label.spark.uniquelabels.downsample;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.label.spark.downsample.MinToInterval;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.CompressionAdapter;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class LabelListDownsampler {

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	private static final String DOWNSAMPLING_FACTORS_KEY = "downsamplingFactors";

	private static final String MULTI_SCALE_KEY = "multiScale";

	public static class CommandLineParameters implements Callable<Void> {

		@Parameters(index = "0", paramLabel = "N5_CONTAINER", description = "Input N5 container")
		private String n5;

		@Parameters(index = "1", paramLabel = "MULTISCALE_GROUP", description = "Multi scale group(relative to N5). Must contain dataset s0 for first scale level.")
		private String multiscaleGroup;

		@Parameters(index = "2", arity = "1..*", paramLabel = "FACTOR", description = "Factor by which to downscale the input image. Factors are relative to the previous level, not to level zero. Format either fx,fy,fz or f")
		private String[] factors;

		@Option(names = {"--block-size", "-b"}, paramLabel = "BLOCK_SIZE", description = "Size of the blocks (in cells) to parallelize with Spark. Format either bx,by,bz or b")
		private String[] blockSize;

		@Option(names = {"-h", "--help"}, usageHelp = true, description = "display a help message")
		private boolean helpRequested;

		public int[][] getFactors() {

			return Arrays.stream(factors).map(LabelListDownsampler::toIntegerArray).toArray(int[][]::new);
		}

		public int[][] getBlockSizes() {

			return Arrays.stream(blockSize).map(LabelListDownsampler::toIntegerArray).toArray(int[][]::new);
		}

		@Override
		public Void call() throws Exception {

			LOG.warn("Will downsample with these factors: {}", Arrays.toString(factors));
			blockSize = blockSize == null ? new String[0] : blockSize;
			final String defaultBlockSize = Arrays.stream(blockSize).limit(factors.length).reduce((f, s) -> s).orElse("64");
			blockSize = Stream.concat(
					Arrays.stream(blockSize).limit(factors.length),
					Stream.generate(() -> defaultBlockSize).limit(factors.length - blockSize.length)).toArray(String[]::new);
			LOG.debug("Block sizes: {}", Arrays.toString(blockSize));
			final int[][] factors = getFactors();
			final int[][] blockSizes = getBlockSizes();

			for (int i = 0; i < factors.length; ++i) {
				if (!checkScaleFactors(factors[i])) {
					LOG.error("Got illegal downscaling factors: {}", factors[i]);
					throw new IllegalArgumentException("Got illegal downscaling factors: " + Arrays.toString(factors[i]));
				}
			}

			final SparkConf conf = new SparkConf().setAppName("SparkDownsampler");
			try (final JavaSparkContext sc = new JavaSparkContext(conf)) {

				donwsampleMultiscale(sc, n5, multiscaleGroup, factors, blockSizes);
			}

			return null;
		}

	}

	public static void run(final String[] args) throws IOException {

		CommandLine.call(new CommandLineParameters(), System.err, args);
	}

	public static void donwsampleMultiscale(
			final JavaSparkContext sc,
			final String n5,
			final String multiscaleGroup,
			final int[][] factors,
			final int[][] blockSizes) throws IOException {

		final N5FSWriter writer = new N5FSWriter(n5);
		addMultiScaleTag(writer, multiscaleGroup);
		final String finestScale = Paths.get(multiscaleGroup, "s0").toString();
		final Map<String, JsonElement> attributeNames = writer.getAttribute(finestScale, "/", JsonObject.class).asMap();
		Arrays.asList("dataType", "compression", "blockSize", "dimensions").forEach(attributeNames::remove);
		for (final Entry<String, JsonElement> entry : attributeNames.entrySet()) {
			writer.setAttribute(multiscaleGroup, entry.getKey(), entry.getValue());
		}

		for (int factorIndex = 0, level = 1; factorIndex < factors.length; ++factorIndex, ++level) {
			final String previousScaleLevel = multiscaleGroup + "/s" + (level - 1);
			final String currentScaleLevel = multiscaleGroup + "/s" + level;

			LabelListDownsampler.downsample(sc,
					new N5FSReader(n5),
					n5,
					previousScaleLevel,
					factors[factorIndex],
					blockSizes[factorIndex],
					n5,
					currentScaleLevel);
		}
	}

	public static void downsample(final JavaSparkContext sc,
			final N5Reader reader,
			final String readGroupName,
			final String readDatasetName,
			final int[] downsampleFactor,
			final int[] blockSize,
			final String outputGroupName,
			final String outputDatasetName) throws IOException {

		final DatasetAttributes attributes = reader.getDatasetAttributes(readDatasetName);

		if (!checkBlockSize(attributes.getBlockSize(), blockSize, downsampleFactor)) {
			LOG.error("Got illegal block sizefactors: previous={}, current={}, factors={}", attributes.getBlockSize(), blockSize, downsampleFactor);
			throw new IllegalArgumentException("Got illegal downscaling factors:");
		}

		final long[] dimensions = attributes.getDimensions();
		final long[] max = Arrays.stream(dimensions).map(dim -> dim - 1).toArray();

		final List<long[]> positions = new ArrayList<>();

		final int nDim = attributes.getNumDimensions();

		final long[] downsampledDimensions = new long[nDim];
		// needs to be cast to long, not int
		Arrays.setAll(downsampledDimensions, i -> (long)Math.ceil((double)dimensions[i] / downsampleFactor[i]));

		final long[] offset = new long[nDim];

		for (int d = 0; d < nDim; ) {

			positions.add(offset.clone());

			for (d = 0; d < nDim; d++) {
				offset[d] += blockSize[d];
				if (offset[d] < downsampledDimensions[d]) {
					break;
				} else {
					offset[d] = 0;
				}
			}
		}

		final N5Writer writer = new N5FSWriter(outputGroupName);
		final double[] previousDownsamplingFactor = Optional
				.ofNullable(reader.getAttribute(readDatasetName, DOWNSAMPLING_FACTORS_KEY, double[].class))
				.orElse(DoubleStream.generate(() -> 1.0).limit(nDim).toArray());
		final double[] accumulatedDownsamplingFactor = IntStream.range(0, nDim).mapToDouble(d -> previousDownsamplingFactor[d] * downsampleFactor[d]).toArray();
		writer.createDataset(outputDatasetName, downsampledDimensions, blockSize, DataType.UINT64, new GzipCompression());
		writer.setAttribute(outputDatasetName, DOWNSAMPLING_FACTORS_KEY, accumulatedDownsamplingFactor);

		sc.parallelize(positions)
				.map(new MinToInterval(max, blockSize))
				.foreach(new LabelListDownsampleFunction(
						readGroupName,
						readDatasetName,
						downsampleFactor,
						outputGroupName,
						outputDatasetName));

		//		System.out.println( "Across " + positions.size() + " parallelized sections, " + output + " cells were downscaled" );
	}

	public static int[] toIntegerArray(final String str) {

		return toIntegerArray(str, 3, ",");
	}

	public static int[] toIntegerArray(final String str, final int requiredNumberOfFields, final String splitRegex) {

		return toIntegerArray(str, new int[requiredNumberOfFields], splitRegex);
	}

	public static int[] toIntegerArray(final String str, final int[] target, final String splitRegex) {

		final String[] split = str.split(splitRegex);
		for (int i = 0; i < split.length; ++i) {
			target[i] = Integer.parseInt(split[i]);
		}
		for (int k = split.length; k < target.length; ++k) {
			target[k] = Integer.parseInt(split[split.length - 1]);
		}

		return target;
	}

	public static Compression fromString(final String str) {

		final GsonBuilder gsonBuilder = new GsonBuilder();
		gsonBuilder.registerTypeHierarchyAdapter(Compression.class, CompressionAdapter.getJsonAdapter());
		final Gson gson = gsonBuilder.create();
		return gson.fromJson(str, Compression.class);
	}

	public static boolean checkScaleFactors(final int[] scaleFactors) {

		for (final int factor : scaleFactors) {
			if (factor < 1) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Each block at lower resolution must border-align with the scaled blocks
	 * at the lower resolution, i.e. the block size of a lower dimension block
	 * multiplied with the scale facotr must be an integer multiple of the block
	 * size at higher resolution for all dimensions.
	 *
	 * @param blockSizePrevious
	 * @param blockSizeCurrent
	 * @param scaleFactors
	 * @return
	 */
	public static boolean checkBlockSize(
			final int[] blockSizePrevious,
			final int[] blockSizeCurrent,
			final int[] scaleFactors) {

		for (int d = 0; d < scaleFactors.length; ++d) {
			if (blockSizeCurrent[d] * scaleFactors[d] % blockSizePrevious[d] != 0) {
				return false;
			}
		}
		return true;
	}

	public static void addMultiScaleTag(final N5Writer n5, final String group) throws IOException {

		if (!n5.exists(group)) {
			n5.createGroup(group);
		}
		n5.setAttribute(group, MULTI_SCALE_KEY, true);
	}
}
