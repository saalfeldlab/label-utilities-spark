package bdv.bigcat.util;

import org.apache.spark.api.java.function.Function;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.label.LabelMultisetType;
import scala.Tuple2;

public class ConvertToDataBlock implements Function< Tuple2< long[], RandomAccessibleInterval< LabelMultisetType > >, ByteArrayDataBlock >
{

	@Override
	public ByteArrayDataBlock call( final Tuple2< long[], RandomAccessibleInterval< LabelMultisetType > > block ) throws Exception
	{
		return HDFConverter.toDataBlock( block._2(), block._1() );
	}

}
