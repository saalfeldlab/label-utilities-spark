package org.janelia.saalfeldlab.label.spark.affinities;

import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.position.FunctionRandomAccessible;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MakeEmptyMask {

	public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {

		final N5FSWriter container = new N5FSWriter("/groups/saalfeld/home/hanslovskyp/data/cremi/sample_A+_padded_20160601-bs=64.n5");
		final String rawPath = "volumes/raw/data/s0";
		final String maskPath = "volumes/masks/raw";

		final RandomAccessibleInterval<UnsignedByteType> raw = N5Utils.open(container, rawPath);
		final DatasetAttributes rawAttributes = container.getDatasetAttributes(rawPath);
		final DatasetAttributes maskAttributes = new DatasetAttributes(
				rawAttributes.getDimensions(),
				rawAttributes.getBlockSize(),
				DataType.UINT8,
				new GzipCompression());

		container.createDataset(maskPath, maskAttributes);
		container.setAttribute(maskPath, "value_range", new double[]{0.0, 1.0});
		container.setAttribute(maskPath, "resolution", new double[]{4, 4, 40});

		final long[] center = new long[maskAttributes.getNumDimensions()];
		Arrays.setAll(center, d -> (raw.min(d) + raw.max(d)) / 2);
		final long[] radius = center.clone();
		final double[] doubleCenterSquared = Arrays.stream(center).asDoubleStream().map(d -> d * d).toArray();

		final RandomAccessible<UnsignedByteType> mask = new FunctionRandomAccessible<>(
				center.length,
				(pos, t) -> {
					double diff = 0;
					for (int d = 0; d < radius.length; ++d) {
						final long diffComponent = pos.getLongPosition(d) - center[d];
						diff += diffComponent * diffComponent / doubleCenterSquared[d];
					}
					t.set(diff < 1 ? 1 : 0);
				},
				UnsignedByteType::new
		);

		final ExecutorService es = Executors.newFixedThreadPool(47);

		N5Utils.save(
				Views.interval(mask, raw),
				container,
				maskPath,
				maskAttributes.getBlockSize(),
				maskAttributes.getCompression(),
				es);

		es.shutdown();

	}

}
