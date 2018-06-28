package org.janelia.saalfeldlab.multisets.spark.exception;

public class InvalidDataset extends Exception
{
	/**
	 *
	 */
	private static final long serialVersionUID = -7931018525983710301L;

	public InvalidDataset( final String datasetIdentifier, final String dataset )
	{
		super( "Invalid dataset for " + datasetIdentifier + ": " + dataset );
	}
}