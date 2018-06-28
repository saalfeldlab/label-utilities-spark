package org.janelia.saalfeldlab.multisets.spark.exception;

public class InvalidN5Container extends Exception
{
	/**
	 *
	 */
	private static final long serialVersionUID = 7933068158292474027L;

	public InvalidN5Container( final String containerIdentifier, final String n5Container )
	{
		super( "N5 container for " + containerIdentifier + " invalid: " + n5Container );
	}
}