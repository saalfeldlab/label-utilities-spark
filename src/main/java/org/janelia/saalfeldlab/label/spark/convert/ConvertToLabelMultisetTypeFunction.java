package org.janelia.saalfeldlab.label.spark.convert;

import org.apache.spark.api.java.function.Function;
import org.janelia.saalfeldlab.label.spark.convert.ConvertToLabelMultisetTypeFunction.ConvertedIntervalWithMaxId;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.type.label.FromIntegerTypeConverter;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.view.Views;

public class ConvertToLabelMultisetTypeFunction< I extends IntegerType< I > > implements
		Function< RandomAccessibleInterval< I >, ConvertedIntervalWithMaxId >
{

	public static class ConvertedIntervalWithMaxId
	{
		public final RandomAccessibleInterval< LabelMultisetType > data;

		public final long maxId;

		public ConvertedIntervalWithMaxId( final RandomAccessibleInterval< LabelMultisetType > data, final long maxId )
		{
			super();
			this.data = data;
			this.maxId = maxId;
		}
	}

	@Override
	public ConvertedIntervalWithMaxId call( final RandomAccessibleInterval< I > data ) throws Exception
	{
		final FromIntegerTypeConverter< I > converter = new FromIntegerTypeConverter<>();
		final LabelMultisetType type = FromIntegerTypeConverter.geAppropriateType();
		long maxId = Long.MIN_VALUE;
		for ( final I i : Views.iterable( data ) )
		{
			final long il = i.getIntegerLong();
			if ( il > maxId )
			{
				maxId = il;
			}
		}
		return new ConvertedIntervalWithMaxId( Converters.convert( data, converter, type ), maxId );
	}

}
