package org.janelia.saalfeldlab.label.spark.convert;

import org.apache.spark.api.java.function.PairFunction;
import org.janelia.saalfeldlab.label.spark.convert.ConvertToLabelMultisetTypeFunction.ConvertedIntervalWithMaxId;

import net.imglib2.util.Intervals;
import scala.Tuple2;

public class AttachBlockPosition implements
		PairFunction< ConvertedIntervalWithMaxId, long[], ConvertedIntervalWithMaxId >
{

	private final int[] blockSize;

	public AttachBlockPosition( final int[] blockSize )
	{
		super();
		this.blockSize = blockSize;
	}

	@Override
	public Tuple2< long[], ConvertedIntervalWithMaxId > call( final ConvertedIntervalWithMaxId interval ) throws Exception
	{
		final long[] pos = ConvertToLabelMultisetType.blockPos( Intervals.minAsLongArray( interval.data ), blockSize );
		return new Tuple2<>( pos, interval );
	}

}
