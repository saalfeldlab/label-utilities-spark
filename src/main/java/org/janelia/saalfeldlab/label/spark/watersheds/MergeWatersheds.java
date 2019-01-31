package org.janelia.saalfeldlab.label.spark.watersheds;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.real.FloatType;

import java.util.function.LongUnaryOperator;

interface MergeWatersheds {

	LongUnaryOperator getMapping(
			RandomAccessibleInterval<FloatType> relief,
			RandomAccessibleInterval<UnsignedLongType> labels,
			long maxId);

}
