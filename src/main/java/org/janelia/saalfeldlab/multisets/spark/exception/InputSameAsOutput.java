package org.janelia.saalfeldlab.multisets.spark.exception;

public class InputSameAsOutput extends Exception
{
	/**
	 *
	 */
	private static final long serialVersionUID = 8009078518484742928L;

	public InputSameAsOutput( final String n5Container, final String dataset )
	{
		super( "Input and output both point to dataset " + dataset + " in container " + n5Container );
	}
}