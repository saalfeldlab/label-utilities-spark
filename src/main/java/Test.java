

import java.io.IOException;
import java.util.Arrays;

import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import gnu.trove.set.hash.TLongHashSet;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

public class Test
{

	public static void main( final String[] args ) throws IOException
	{
		final String n5Image = "/home/phil/Downloads/sample_A_padded_20160501.hdf";
		final String n5Label = "/home/phil/local/tmp/sample_A_padded_20160501-argmax.n5";
		final String datasetImage = "volumes/labels/neuron_ids";

		final TLongHashSet labelsFromImage = new TLongHashSet();

		final RandomAccessibleInterval< UnsignedLongType > image = N5Utils.open( new N5HDF5Reader( n5Image, 64, 64, 64 ), datasetImage );
		Views.iterable( image ).forEach( px -> labelsFromImage.add( px.getIntegerLong() ) );

		final N5Reader reader = new N5FSReader( n5Label );

		for ( int i = 0; i < 5; ++i )
		{
			System.out.println( "level " + i );
			final TLongHashSet labelsFromLabel = new TLongHashSet();
			final String datasetLabel = "volumes/labels/neuron_ids/unique-labels-per-block/s" + i;
			final DatasetAttributes attrs = reader.getDatasetAttributes( datasetLabel );
			final long[] dims = attrs.getDimensions();
			final int[] blockSize = attrs.getBlockSize();

			if ( i == 0 )
			{
				if ( !Arrays.equals( dims, new N5HDF5Reader( n5Image, 64, 64, 64 ).getDatasetAttributes( datasetImage ).getDimensions() ) )
				{
					System.out.println( "DIMENSIONALITY MISMATCH!" );
				}
			}

			for ( final Interval interval : Grids.collectAllContainedIntervals( dims, blockSize ) )
			{
				final long[] block = Intervals.minAsLongArray( interval );
				Arrays.setAll( block, d -> block[ d ] / blockSize[ d ] );
				final LongArrayDataBlock dataBlock = ( LongArrayDataBlock ) reader.readBlock( datasetLabel, attrs, block );

				if ( i == 0 )
				{
					final TLongHashSet localUnique = new TLongHashSet();
					for ( final UnsignedLongType ult : Views.interval( image, interval ) )
					{
						localUnique.add( ult.getIntegerLong() );
					}
					if ( !localUnique.equals( new TLongHashSet( dataBlock.getData() ) ) )
					{
						System.out.print( "BLOCK MISMATCH FOR " + Arrays.toString( Intervals.minAsLongArray( interval ) ) );
					}
				}

				labelsFromLabel.addAll( dataBlock.getData() );
			}
			if ( !labelsFromLabel.equals( labelsFromImage ) )
			{
				System.out.println( "WHY DIFFERENT ? " );
				System.out.println( labelsFromLabel );
				System.out.print( labelsFromImage );
			}
		}
		System.out.println( "Done!" );

	}

}
