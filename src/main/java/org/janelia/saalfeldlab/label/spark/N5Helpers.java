package org.janelia.saalfeldlab.label.spark;

import org.janelia.saalfeldlab.n5.N5Exception;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.universe.N5Factory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.Optional;
import java.util.regex.Pattern;

public class N5Helpers {

	public static final String LABEL_MULTISETTYPE_KEY = "isLabelMultiset";

	public static final int DEFAULT_BLOCK_SIZE = 64;

	public static final String MULTISCALE_KEY = "multiScale";

	public static final String MAX_ID_KEY = "maxId";

	private static final Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

	public static N5Reader n5Reader(final String base, final int... defaultCellDimensions) throws IOException {

		final var factory = new N5Factory();
		factory.hdf5DefaultBlockSize(defaultCellDimensions);
		return factory.openReader(base);
	}

	public static N5Writer n5Writer(final String base, final int... defaultCellDimensions) throws IOException {

		final var factory = new N5Factory();
		factory.hdf5DefaultBlockSize(defaultCellDimensions);
		return factory.openWriter(base);
	}

	public static boolean isHDF(final String base) {

		LOG.debug("Checking {} for HDF", base);
		final boolean isHDF = Pattern.matches("^h5://", base) || Pattern.matches("^.*\\.(hdf|h5)$", base);
		LOG.debug("{} is hdf5? {}", base, isHDF);
		return isHDF;
	}

	public static long[] blockPos(final long[] position, final int[] blockSize) {

		final long[] blockPos = new long[position.length];
		Arrays.setAll(blockPos, d -> position[d] / blockSize[d]);
		return blockPos;
	}

	public static boolean isMultiScale(final N5Reader reader, final String dataset) throws IOException {

		return Optional.ofNullable(reader.getAttribute(dataset, MULTISCALE_KEY, Boolean.class)).orElse(false);
	}

	public static String[] listScaleDatasets(final N5Reader n5, final String group) throws IOException {

		final String[] scaleDirs = Arrays
				.stream(n5.list(group))
				.filter(s -> s.matches("^s\\d+$"))
				.filter(s -> {
					try {
						return n5.datasetExists(group + "/" + s);
					} catch (final N5Exception e) {
						return false;
					}
				})
				.toArray(String[]::new);

		LOG.debug("Found these scale dirs: {}", Arrays.toString(scaleDirs));
		return scaleDirs;
	}

	public static String[] listAndSortScaleDatasets(final N5Reader n5, final String group) throws IOException {

		final String[] scaleDirs = listScaleDatasets(n5, group);
		sortScaleDatasets(scaleDirs);

		LOG.debug("Sorted scale dirs: {}", Arrays.toString(scaleDirs));
		return scaleDirs;
	}

	public static void sortScaleDatasets(final String[] scaleDatasets) {

		Arrays.sort(scaleDatasets, (f1, f2) -> {
			return Integer.compare(
					Integer.parseInt(f1.replaceAll("[^\\d]", "")),
					Integer.parseInt(f2.replaceAll("[^\\d]", "")));
		});
	}

	public static <T> T revertInplaceAndReturn(final T t, final boolean revert) {

		if (!revert) {
			return t;
		}

		if (t instanceof boolean[]) {
			final boolean[] arr = (boolean[])t;
			for (int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k) {
				final boolean v = arr[0];
				arr[0] = arr[k];
				arr[k] = v;
			}
		}

		if (t instanceof byte[]) {
			final byte[] arr = (byte[])t;
			for (int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k) {
				final byte v = arr[0];
				arr[0] = arr[k];
				arr[k] = v;
			}
		}

		if (t instanceof char[]) {
			final char[] arr = (char[])t;
			for (int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k) {
				final char v = arr[0];
				arr[0] = arr[k];
				arr[k] = v;
			}
		}

		if (t instanceof short[]) {
			final short[] arr = (short[])t;
			for (int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k) {
				final short v = arr[0];
				arr[0] = arr[k];
				arr[k] = v;
			}
		}

		if (t instanceof int[]) {
			final int[] arr = (int[])t;
			for (int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k) {
				final int v = arr[0];
				arr[0] = arr[k];
				arr[k] = v;
			}
		}

		if (t instanceof long[]) {
			final long[] arr = (long[])t;
			for (int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k) {
				final long v = arr[0];
				arr[0] = arr[k];
				arr[k] = v;
			}
		}

		if (t instanceof float[]) {
			final float[] arr = (float[])t;
			for (int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k) {
				final float v = arr[0];
				arr[0] = arr[k];
				arr[k] = v;
			}
		}

		if (t instanceof double[]) {
			final double[] arr = (double[])t;
			for (int i = 0, k = arr.length - 1; i < arr.length / 2; ++i, --k) {
				final double v = arr[0];
				arr[0] = arr[k];
				arr[k] = v;
			}
		}

		return t;
	}
}
