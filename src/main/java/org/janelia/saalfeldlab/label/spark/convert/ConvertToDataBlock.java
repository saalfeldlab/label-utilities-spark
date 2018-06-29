package org.janelia.saalfeldlab.label.spark.convert;

import org.apache.spark.api.java.function.PairFunction;
import org.janelia.saalfeldlab.label.spark.convert.ConvertToLabelMultisetTypeFunction.ConvertedIntervalWithMaxId;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;

import scala.Tuple2;

public class ConvertToDataBlock implements PairFunction< Tuple2< long[], ConvertedIntervalWithMaxId >, ByteArrayDataBlock, Long >
{

	@Override
	public Tuple2< ByteArrayDataBlock, Long > call( final Tuple2< long[], ConvertedIntervalWithMaxId > block ) throws Exception
	{
		return new Tuple2<>( ConvertToLabelMultisetType.toDataBlock( block._2().data, block._1() ), block._2().maxId );
	}

}
