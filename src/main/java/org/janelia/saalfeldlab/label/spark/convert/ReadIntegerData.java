package org.janelia.saalfeldlab.label.spark.convert;

import java.io.IOException;

import org.apache.spark.api.java.function.Function;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.view.Views;

public class ReadIntegerData< I extends IntegerType< I > & NativeType< I > > implements Function< Interval, RandomAccessibleInterval< I > >
{

	private final String group;

	private final String dataset;

	private final int[] defaultBlockSize;

	public ReadIntegerData( final String group, final String dataset, final int[] defaultBlockSize )
	{
		super();
		this.group = group;
		this.dataset = dataset;
		this.defaultBlockSize = defaultBlockSize;
	}

	@Override
	public RandomAccessibleInterval< I > call( final Interval interval )
	{
		try
		{
			return Views.interval(
					( RandomAccessibleInterval< I > ) N5Utils.open( ConvertToLabelMultisetType.n5Reader( group, defaultBlockSize ), dataset ),
					interval );
		}
		catch ( final IOException e )
		{
			throw new RuntimeException( e );
		}
	}

}
