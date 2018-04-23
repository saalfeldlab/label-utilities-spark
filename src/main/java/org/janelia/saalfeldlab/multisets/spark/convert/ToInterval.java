package org.janelia.saalfeldlab.multisets.spark.convert;

import java.util.Arrays;

import org.apache.spark.api.java.function.Function;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;

public class ToInterval implements Function< long[], Interval >
{

	private final long[] dim;

	private final int[] blockSize;

	public ToInterval( final long[] dim, final int[] blockSize )
	{
		super();
		this.dim = dim;
		this.blockSize = blockSize;
	}

	@Override
	public Interval call( final long[] min ) throws Exception
	{
		final long[] max = new long[ dim.length ];
		Arrays.setAll( max, d -> Math.min( min[ d ] + blockSize[ d ], dim[ d ] ) - 1 );
		return new FinalInterval( min, max );
	}

}
