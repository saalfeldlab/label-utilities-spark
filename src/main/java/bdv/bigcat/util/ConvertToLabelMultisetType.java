package bdv.bigcat.util;

import org.apache.spark.api.java.function.Function;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.type.label.FromIntegerTypeConverter;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.numeric.IntegerType;

public class ConvertToLabelMultisetType< I extends IntegerType< I > > implements Function< RandomAccessibleInterval< I >, RandomAccessibleInterval< LabelMultisetType > >
{

	@Override
	public RandomAccessibleInterval< LabelMultisetType > call( final RandomAccessibleInterval< I > data ) throws Exception
	{
		final FromIntegerTypeConverter< I > converter = new FromIntegerTypeConverter<>();
		final LabelMultisetType type = FromIntegerTypeConverter.geAppropriateType();
		return Converters.convert( data, converter, type );
	}

}
