package org.janelia.saalfeldlab.label.spark.downsample;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import org.apache.spark.api.java.function.Function;

import java.util.Arrays;

public class MinToInterval implements Function<long[], Interval> {

	private final long[] max;

	private final int[] blockSize;

	public MinToInterval(final long[] max, final int[] blockSize) {

		super();
		this.max = max;
		this.blockSize = blockSize;
	}

	@Override
	public Interval call(final long[] min) {

		final long[] max = new long[min.length];
		Arrays.setAll(max, d -> Math.min(min[d] + blockSize[d] - 1, this.max[d]));
		return new FinalInterval(min, max);
	}

}
