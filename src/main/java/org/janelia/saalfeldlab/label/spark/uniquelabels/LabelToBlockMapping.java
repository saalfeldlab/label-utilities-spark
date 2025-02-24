package org.janelia.saalfeldlab.label.spark.uniquelabels;

import com.google.gson.GsonBuilder;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.algorithm.util.Singleton;
import net.imglib2.util.Intervals;
import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.message.BasicNameValuePair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.label.spark.N5Helpers;
import org.janelia.saalfeldlab.label.spark.exception.InvalidDataset;
import org.janelia.saalfeldlab.label.spark.exception.InvalidN5Container;
import org.janelia.saalfeldlab.labels.blocks.LabelBlockLookup;
import org.janelia.saalfeldlab.labels.blocks.LabelBlockLookupAdapter;
import org.janelia.saalfeldlab.labels.blocks.n5.LabelBlockLookupFromN5Relative;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sparkproject.guava.io.Files;
import picocli.CommandLine;
import picocli.CommandLine.Parameters;
import scala.Tuple2;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.lang.invoke.MethodHandles;
import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class LabelToBlockMapping {

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	public static class CommandLineParameters implements Callable<Void> {

		@Parameters(index = "0", paramLabel = "INPUT_N5", description = "Input N5 container")
		private String inputN5;

		@Parameters(index = "1", paramLabel = "INPUT_DATASET", description = "Input dataset: contained labels in each block. Must deserialize to long[] array.")
		private String inputDataset;

		@Parameters(index = "2", paramLabel = "OUTPUT_DIRECTORY", description = "Output directory. If --store-as-n5 flag is used, this is a group in the container OUTPUT_N5")
		private String outputDirectory;

		@CommandLine.Option(names = {"--store-as-n5"}, paramLabel = "N5_STEP_SIZE", description = "Store as n5 instead of single file with <N5_STEP_SIZE> entries per block.")
		private Integer n5StepSize;

		@CommandLine.Option(names = {"--output-n5"}, paramLabel = "OUTPUT_N5", description = "Only used with `--store-as-n5' flag. Defaults to INPUT_N5 if not specified")
		private String outputN5 = null;

		@Override
		public Void call() throws Exception {

			if (this.inputN5 == null) {
				throw new InvalidN5Container("INPUT_N5", this.inputN5);
			}

			if (this.inputDataset == null) {
				throw new InvalidDataset("INPUT_DATASET", inputDataset);
			}

			this.outputN5 = this.outputN5 == null ? this.inputN5 : this.outputN5;

			final long startTime = System.currentTimeMillis();

			final SparkConf conf = new SparkConf().setAppName(MethodHandles.lookup().lookupClass().getName());

			try (final JavaSparkContext sc = new JavaSparkContext(conf)) {
				if (this.n5StepSize == null)
					createMappingWithMultiscaleCheck(sc, inputN5, inputDataset, outputDirectory);
				else // TODO what to do about relativeLookupGroup?
					createMappingWithMultiscaleCheckN5(sc, inputN5, inputDataset, outputN5, outputDirectory, null, this.n5StepSize);
			}

			final long endTime = System.currentTimeMillis();

			final String formattedTime = DurationFormatUtils.formatDuration(endTime - startTime, "HH:mm:ss.SSS");
			LOG.info("Created label to block mapping for " + inputN5 + ":" + inputDataset + " at " + outputDirectory + " in " + formattedTime);
			return null;

		}
	}

	public static void run(final String[] args) throws IOException {

		CommandLine.call(new CommandLineParameters(), System.err, args);
	}

	public static void createMappingWithMultiscaleCheckN5(
			final JavaSparkContext sc,
			final String inputN5,
			final String inputDataset,
			final String outputN5,
			final String enclosingGroup,
			final String relativeLookupGroup,
			final int stepSize) throws IOException {

		final N5Reader reader = N5Helpers.n5Reader(inputN5, N5Helpers.DEFAULT_BLOCK_SIZE);
		final N5Factory factory = N5Helpers.defaultFactory();
		factory.gsonBuilder(new GsonBuilder().registerTypeHierarchyAdapter(LabelBlockLookup.class, LabelBlockLookupAdapter.getJsonAdapter()));
		final N5Writer writer = factory.openWriter(outputN5);

		final String groupPath = relativeLookupGroup == null ? enclosingGroup : enclosingGroup + "/" + relativeLookupGroup;
		final String pattern = relativeLookupGroup == null ? "s%d" : relativeLookupGroup + "/s%d";
		final LabelBlockLookupN5Supplier lookupSupplier = new LabelBlockLookupN5Supplier(outputN5, enclosingGroup, pattern);
		writer.setAttribute(enclosingGroup, "labelBlockLookup", lookupSupplier.get());
		if (!writer.exists(groupPath))
			writer.createGroup(groupPath);

		if (N5Helpers.isMultiScale(reader, inputDataset)) {
			final String[] sortedScaleDirs = N5Helpers.listAndSortScaleDatasets(reader, inputDataset);
			for (int level = 0; level < sortedScaleDirs.length; ++level) {
				final String scaleDataset = sortedScaleDirs[level];
				//				writer.createDataset( enclosingGroup + "/" + scaleDataset, new long[] { Long.MAX_VALUE }, new int[] { stepSize }, DataType.INT8, new GzipCompression() );
				writer.createDataset(String.format(enclosingGroup + "/" + pattern, level), new long[]{Long.MAX_VALUE}, new int[]{stepSize}, DataType.INT8, new GzipCompression());
				LOG.info("Creating mapping for scale dataset {} in group {} of n5 container {} at target {}", scaleDataset, inputDataset, inputN5, enclosingGroup);
				createMappingN5(
						sc,
						inputN5,
						inputDataset + "/" + scaleDataset,
						lookupSupplier,
						level,
						stepSize);
			}
		} else {
			createMappingN5(
					sc,
					inputN5,
					inputDataset,
					lookupSupplier,
					0,
					stepSize);
		}
	}

	public static void createMappingWithMultiscaleCheck(
			final JavaSparkContext sc,
			final String inputN5,
			final String inputDataset,
			final String outputDirectory) throws IOException {

		final N5Reader reader = N5Helpers.n5Reader(inputN5, N5Helpers.DEFAULT_BLOCK_SIZE);
		if (N5Helpers.isMultiScale(reader, inputDataset)) {
			for (final String scaleDataset : N5Helpers.listScaleDatasets(reader, inputDataset)) {
				LOG.info("Creating mapping for scale dataset {} in group {} of n5 container {} at target {}", scaleDataset, inputDataset, inputN5, outputDirectory);
				createMapping(sc, inputN5, inputDataset + "/" + scaleDataset, Paths.get(outputDirectory, scaleDataset).toAbsolutePath().toString());
			}
		} else {
			LOG.info("Creating mapping for dataset {} of n5 container {} at target {}", inputDataset, inputN5, outputDirectory);
			createMapping(sc, inputN5, inputDataset, outputDirectory);
		}
	}

	public static void createMappingN5(
			final JavaSparkContext sc,
			final String inputN5,
			final String inputDataset,
			final Supplier<LabelBlockLookupFromN5Relative> blockLookup,
			final int level,
			final int stepSize) throws IOException {

		final N5Reader reader = N5Helpers.n5Reader(inputN5);
		final DatasetAttributes inputAttributes = reader.getDatasetAttributes(inputDataset);
		final int[] blockSize = inputAttributes.getBlockSize();
		final long[] dims = inputAttributes.getDimensions();
		final List<Tuple2<long[], long[]>> intervals = Grids
				.collectAllContainedIntervals(dims, blockSize)
				.stream()
				.map(i -> new Tuple2<>(Intervals.minAsLongArray(i), Intervals.maxAsLongArray(i)))
				.collect(Collectors.toList());
		sc
				.parallelize(intervals)
				.mapToPair(minMax -> {
					final long[] blockPos = N5Helpers.blockPos(minMax._1(), blockSize);
					final String readerCacheKey = new URIBuilder(StorageFormat.parseUri(inputN5).getB()).setParameters(
							new BasicNameValuePair("type", "reader"),
							new BasicNameValuePair("call", "label-block-mapping-create-mapping-n5")
					).toString();
					final N5Reader n5reader = Singleton.get(readerCacheKey, () -> N5Helpers.n5Reader(inputN5, N5Helpers.DEFAULT_BLOCK_SIZE));
					final LongArrayDataBlock block = (LongArrayDataBlock)n5reader.readBlock(inputDataset, new DatasetAttributes(dims, blockSize, DataType.UINT64, new GzipCompression()), blockPos);
					return new Tuple2<>(minMax, block.getData());
				})
				.flatMapToPair(input -> Arrays
						.stream(input._2())
						.mapToObj(id -> new Tuple2<>(id, input._1()))
						.iterator())
				.aggregateByKey(
						new ArrayList<Tuple2<long[], long[]>>(),
						(l, v) -> {
							l.add(v);
							return l;
						},
						(al1, al2) -> {
							final ArrayList<Tuple2<long[], long[]>> al = new ArrayList<>();
							al.addAll(al1);
							al.addAll(al2);
							return al;
						})
				.mapToPair(input -> new Tuple2<>(input._1() / stepSize, input))
				.aggregateByKey(
						new ArrayList<Tuple2<Long, ArrayList<Tuple2<long[], long[]>>>>(),
						(l, v) -> {
							l.add(v);
							return l;
						},
						(al1, al2) -> {
							final ArrayList<Tuple2<Long, ArrayList<Tuple2<long[], long[]>>>> al = new ArrayList<>();
							al.addAll(al1);
							al.addAll(al2);
							return al;
						})
				.values()
				.foreach(list -> {
					final LabelBlockLookupFromN5Relative bl = blockLookup.get();
					final HashMap<Long, Interval[]> map = new HashMap<>();
					list.stream().map(t -> new Tuple2<>(t._1(), t._2().stream().map(p -> new FinalInterval(p._1(), p._2())).toArray(Interval[]::new))).forEach(t -> map.put(t._1(), t._2()));
					bl.set(level, map);
				});
	}

	public static void createMapping(
			final JavaSparkContext sc,
			final String inputN5,
			final String inputDataset,
			final String outputDirectory) throws IOException {

		final N5Reader reader = N5Helpers.n5Reader(inputN5);
		final DatasetAttributes inputAttributes = reader.getDatasetAttributes(inputDataset);
		final int[] blockSize = inputAttributes.getBlockSize();
		final long[] dims = inputAttributes.getDimensions();
		new File(outputDirectory).mkdirs();
		final List<Tuple2<long[], long[]>> intervals = Grids
				.collectAllContainedIntervals(dims, blockSize)
				.stream()
				.map(i -> new Tuple2<>(Intervals.minAsLongArray(i), Intervals.maxAsLongArray(i)))
				.collect(Collectors.toList());
		sc
				.parallelize(intervals)
				.mapToPair(minMax -> {
					final long[] blockPos = N5Helpers.blockPos(minMax._1(), blockSize);
					final String readerCacheKey = new URIBuilder(StorageFormat.parseUri(inputN5).getB()).setParameters(
							new BasicNameValuePair("type", "reader"),
							new BasicNameValuePair("call", "label-block-mapping-create-mapping")
					).toString();
					final N5Reader n5reader = Singleton.get(readerCacheKey, () -> N5Helpers.n5Reader(inputN5, N5Helpers.DEFAULT_BLOCK_SIZE));
					final LongArrayDataBlock block = (LongArrayDataBlock)n5reader.readBlock(inputDataset, new DatasetAttributes(dims, blockSize, DataType.UINT64, new GzipCompression()), blockPos);
					return new Tuple2<>(minMax, block.getData());
				})
				.flatMapToPair(input -> Arrays
						.stream(input._2())
						.mapToObj(id -> new Tuple2<>(id, input._1()))
						.iterator())
				.aggregateByKey(
						new ArrayList<Tuple2<long[], long[]>>(),
						(l, v) -> {
							l.add(v);
							return l;
						},
						(al1, al2) -> {
							final ArrayList<Tuple2<long[], long[]>> al = new ArrayList<>();
							al.addAll(al1);
							al.addAll(al2);
							return al;
						})
				.foreach(list -> {
					final String id = list._1().toString();
					// 2 (min, max) * 3 (three dimensions) values of size
					// Long.BYTES per list entry
					final int numBytes = 2 * 3 * list._2().size() * Long.BYTES;
					final byte[] data = new byte[numBytes];
					final ByteBuffer bb = ByteBuffer.wrap(data);
					for (final Tuple2<long[], long[]> t : list._2()) {
						final long[] l1 = t._1();
						final long[] l2 = t._2();
						bb.putLong(l1[0]);
						bb.putLong(l1[1]);
						bb.putLong(l1[2]);
						bb.putLong(l2[0]);
						bb.putLong(l2[1]);
						bb.putLong(l2[2]);
					}
					Files.write(data, Paths.get(outputDirectory, id).toFile());
				});
	}

	private static class LabelBlockLookupN5Supplier implements Supplier<LabelBlockLookupFromN5Relative>, Serializable {

		private final String root;

		private final String group;

		private final String pattern;

		private LabelBlockLookupN5Supplier(
				final String root,
				final String group,
				final String pattern) {

			this.root = root;
			this.group = group;
			this.pattern = pattern;
		}

		@Override
		public LabelBlockLookupFromN5Relative get() {

			final LabelBlockLookupFromN5Relative lookup = new LabelBlockLookupFromN5Relative(pattern);
			lookup.setRelativeTo(N5Helpers.n5Writer(root), group);
			return lookup;
		}
	}

}
