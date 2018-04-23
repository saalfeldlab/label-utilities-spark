package org.janelia.saalfeldlab.multisets.spark.convert;

import org.apache.spark.api.java.function.PairFunction;

import net.imglib2.Interval;
import net.imglib2.util.Intervals;
import scala.Tuple2;

public class AttachBlockPosition< I extends Interval > implements PairFunction< I, long[], I >
{

	private final int[] blockSize;

	public AttachBlockPosition( final int[] blockSize )
	{
		super();
		this.blockSize = blockSize;
	}

	@Override
	public Tuple2< long[], I > call( final I interval ) throws Exception
	{
		final long[] pos = ConvertToLabelMultisetType.blockPos( Intervals.minAsLongArray( interval ), blockSize );
		return new Tuple2<>( pos, interval );
	}

}
