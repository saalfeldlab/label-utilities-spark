package org.janelia.saalfeldlab.label.spark;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.Optional;
import java.util.regex.Pattern;

import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Writer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class N5Helpers
{

	public static final String LABEL_MULTISETTYPE_KEY = "isLabelMultiset";

	public static final int DEFAULT_BLOCK_SIZE = 64;

	public static final String MULTISCALE_KEY = "multiScale";

	public static final String MAX_ID_KEY = "maxId";

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static N5Reader n5Reader( final String base, final int... defaultCellDimensions ) throws IOException
	{
		return isHDF( base ) ? new N5HDF5Reader( base, defaultCellDimensions ) : new N5FSReader( base );
	}

	public static N5Writer n5Writer( final String base, final int... defaultCellDimensions ) throws IOException
	{
		return isHDF( base ) ? new N5HDF5Writer( base, defaultCellDimensions ) : new N5FSWriter( base );
	}

	public static boolean isHDF( final String base )
	{
		LOG.debug( "Checking {} for HDF", base );
		final boolean isHDF = Pattern.matches( "^h5://", base ) || Pattern.matches( "^.*\\.(hdf|h5)$", base );
		LOG.debug( "{} is hdf5? {}", base, isHDF );
		return isHDF;
	}

	public static long[] blockPos( final long[] position, final int[] blockSize )
	{
		final long[] blockPos = new long[ position.length ];
		Arrays.setAll( blockPos, d -> position[ d ] / blockSize[ d ] );
		return blockPos;
	}

	public static boolean isMultiScale( final N5Reader reader, final String dataset ) throws IOException
	{
		return Optional.ofNullable( reader.getAttribute( dataset, MULTISCALE_KEY, Boolean.class ) ).orElse( false );
	}

	public static String[] listScaleDatasets( final N5Reader n5, final String group ) throws IOException
	{
		final String[] scaleDirs = Arrays
				.stream( n5.list( group ) )
				.filter( s -> s.matches( "^s\\d+$" ) )
				.filter( s -> {
					try
					{
						return n5.datasetExists( group + "/" + s );
					}
					catch ( final IOException e )
					{
						return false;
					}
				} )
				.toArray( String[]::new );

		LOG.debug( "Found these scale dirs: {}", Arrays.toString( scaleDirs ) );
		return scaleDirs;
	}

	public static String[] listAndSortScaleDatasets( final N5Reader n5, final String group ) throws IOException
	{
		final String[] scaleDirs = listScaleDatasets( n5, group );
		sortScaleDatasets( scaleDirs );

		LOG.debug( "Sorted scale dirs: {}", Arrays.toString( scaleDirs ) );
		return scaleDirs;
	}

	public static void sortScaleDatasets( final String[] scaleDatasets )
	{
		Arrays.sort( scaleDatasets, ( f1, f2 ) -> {
			return Integer.compare(
					Integer.parseInt( f1.replaceAll( "[^\\d]", "" ) ),
					Integer.parseInt( f2.replaceAll( "[^\\d]", "" ) ) );
		} );
	}
}
