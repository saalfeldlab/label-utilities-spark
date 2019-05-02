package org.janelia.saalfeldlab.label.spark.watersheds;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedLongType;

import java.util.function.LongUnaryOperator;

interface MergeWatersheds {

	<T extends RealType<T>> LongUnaryOperator getMapping(
			RandomAccessibleInterval<T> relief,
			RandomAccessibleInterval<UnsignedLongType> labels,
			long maxId);

}
