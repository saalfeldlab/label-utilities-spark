package org.janelia.saalfeldlab.label.spark.exception;

import org.janelia.saalfeldlab.n5.DataType;

public class InvalidDataType extends Exception
{
	/**
	 *
	 */
	private static final long serialVersionUID = 3108292922725401520L;

	public InvalidDataType( final DataType dataType )
	{
		super( "Only (unsigned) integer or LabelMultisetType data supported. Got: " + dataType );
	}
}