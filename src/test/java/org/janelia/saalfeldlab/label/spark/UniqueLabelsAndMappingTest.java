package org.janelia.saalfeldlab.label.spark;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.label.spark.exception.InputSameAsOutput;
import org.janelia.saalfeldlab.label.spark.exception.InvalidDataType;
import org.janelia.saalfeldlab.label.spark.exception.InvalidDataset;
import org.janelia.saalfeldlab.label.spark.exception.InvalidN5Container;
import org.janelia.saalfeldlab.label.spark.uniquelabels.ExtractUniqueLabelsPerBlock;
import org.janelia.saalfeldlab.label.spark.uniquelabels.LabelToBlockMapping;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.RawCompression;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import net.imglib2.AbstractInterval;
import net.imglib2.Interval;
import net.imglib2.algorithm.util.Grids;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.util.Intervals;

public class UniqueLabelsAndMappingTest
{

	private final String labelDataset = "labels";

	private final String uniqueLabelDataset = "unique-labels";

	private final String labelToBlocksMappingDirectory;

	private final String tmpDir;

	private final long[] dims = { 4, 3, 2 };

	private final int[] blockSize = { 3, 2, 2 };

	private final long[] labels = {
			1, 1, 1, 1,
			1, 1, 2, 2,
			1, 2, 2, 2,

			1, 1, 2, 4,
			3, 3, 2, 2,
			3, 3, 3, 2
	};

	ArrayImg< UnsignedLongType, LongArray > labelImg = ArrayImgs.unsignedLongs( labels, dims );

	Map< String, Set< ComparableFinalInterval > > groundTruthMapping = new HashMap<>();
	{
		groundTruthMapping.put( "1", new HashSet<>( Arrays.asList(
				new ComparableFinalInterval( new long[] { 0, 0, 0 }, new long[] { 2, 1, 1 } ),
				new ComparableFinalInterval( new long[] { 3, 0, 0 }, new long[] { 3, 1, 1 } ),
				new ComparableFinalInterval( new long[] { 0, 2, 0 }, new long[] { 2, 2, 1 } ) ) ) );

		groundTruthMapping.put( "2", new HashSet<>( Arrays.asList(
				new ComparableFinalInterval( new long[] { 0, 0, 0 }, new long[] { 2, 1, 1 } ),
				new ComparableFinalInterval( new long[] { 3, 0, 0 }, new long[] { 3, 1, 1 } ),
				new ComparableFinalInterval( new long[] { 0, 2, 0 }, new long[] { 2, 2, 1 } ),
				new ComparableFinalInterval( new long[] { 3, 2, 0 }, new long[] { 3, 2, 1 } ) ) ) );

		groundTruthMapping.put( "3", new HashSet<>( Arrays.asList(
				new ComparableFinalInterval( new long[] { 0, 0, 0 }, new long[] { 2, 1, 1 } ),
				new ComparableFinalInterval( new long[] { 0, 2, 0 }, new long[] { 2, 2, 1 } ) ) ) );

		groundTruthMapping.put( "4", new HashSet<>( Arrays.asList(
				new ComparableFinalInterval( new long[] { 3, 0, 0 }, new long[] { 3, 1, 1 } ) ) ) );

	}

	public UniqueLabelsAndMappingTest() throws IOException
	{
		this.tmpDir = Files.createTempDirectory( "unique-labels-test" ).toAbsolutePath().toString();
		this.labelToBlocksMappingDirectory = Paths.get( tmpDir, "label-to-block-mapping" ).toAbsolutePath().toString();
	}

	@Before
	public void setUp() throws IOException
	{

		final N5FSWriter n5 = new N5FSWriter( this.tmpDir );
		n5.createDataset( labelDataset, new DatasetAttributes( dims, blockSize, DataType.UINT64, new RawCompression() ) );
		N5Utils.save( labelImg, n5, labelDataset, blockSize, new RawCompression() );
	}

	@After
	public void tearDown() throws IOException
	{
		FileUtils.deleteDirectory( new File( this.tmpDir ) );
	}

	@Test
	public void test() throws InvalidDataType, IOException, InvalidN5Container, InvalidDataset, InputSameAsOutput
	{
		final SparkConf conf = new SparkConf()
				.setAppName( getClass().getName() )
				.setMaster( "local[*]" );
		try (JavaSparkContext sc = new JavaSparkContext( conf ))
		{
			ExtractUniqueLabelsPerBlock.extractUniqueLabels( sc, tmpDir, tmpDir, labelDataset, uniqueLabelDataset );
			LabelToBlockMapping.createMapping( sc, tmpDir, uniqueLabelDataset, labelToBlocksMappingDirectory );
		}
		final N5Reader n5 = new N5FSReader( tmpDir );
		final DatasetAttributes uniqueLabelAttributes = n5.getDatasetAttributes( uniqueLabelDataset );

		Assert.assertArrayEquals( dims, uniqueLabelAttributes.getDimensions() );
		Assert.assertArrayEquals( blockSize, uniqueLabelAttributes.getBlockSize() );

		final List< long[] > blocks = Grids.collectAllOffsets( dims, blockSize );
		for ( final long[] block : blocks )
		{
			final long[] blockPosition = block.clone();
			Arrays.setAll( blockPosition, d -> blockPosition[ d ] / blockSize[ d ] );

			final LongArrayDataBlock blockData = ( ( LongArrayDataBlock ) n5.readBlock( uniqueLabelDataset, uniqueLabelAttributes, blockPosition ) );
			final long[] sortedContents = blockData.getData().clone();
			Arrays.sort( sortedContents );

			if ( Arrays.equals( blockPosition, new long[] { 0, 0, 0 } ) )
			{
				Assert.assertArrayEquals( new long[] { 1, 2, 3 }, sortedContents );
			}

			else if ( Arrays.equals( blockPosition, new long[] { 1, 0, 0 } ) )
			{
				Assert.assertArrayEquals( new long[] { 1, 2, 4 }, sortedContents );
			}

			else if ( Arrays.equals( blockPosition, new long[] { 0, 1, 0 } ) )
			{
				Assert.assertArrayEquals( new long[] { 1, 2, 3 }, sortedContents );
			}

			else if ( Arrays.equals( blockPosition, new long[] { 1, 1, 0 } ) )
			{
				Assert.assertArrayEquals( new long[] { 2 }, sortedContents );
			}

			else
			{
				Assert.fail( "Observed unexpected block position: " + Arrays.toString( blockPosition ) );
			}
		}

		final String[] containedFiles = new File( labelToBlocksMappingDirectory ).list();
		Arrays.sort( containedFiles );
		Assert.assertEquals( 4, containedFiles.length );
		final Set< String > containedFilesSet = new HashSet<>( Arrays.asList( containedFiles ) );
		Assert.assertEquals( new HashSet<>( Arrays.asList( "1", "2", "3", "4" ) ), containedFilesSet );
		for ( final String file : containedFiles )
		{
			final Set< ComparableFinalInterval > storedIntervals = new HashSet<>();
			final byte[] data = Files.readAllBytes( Paths.get( labelToBlocksMappingDirectory, file ) );
			// three dimensions, 2 arrays, long elements
			Assert.assertEquals( 0, data.length % ( 3 * 2 * Long.BYTES ) );
			final ByteBuffer bb = ByteBuffer.wrap( data );
			while ( bb.hasRemaining() )
			{
				storedIntervals.add( new ComparableFinalInterval(
						new long[] { bb.getLong(), bb.getLong(), bb.getLong() },
						new long[] { bb.getLong(), bb.getLong(), bb.getLong() } ) );
			}
			Assert.assertEquals( groundTruthMapping.get( file ), storedIntervals );
		}

	}

	private static final class ComparableFinalInterval extends AbstractInterval
	{

		public ComparableFinalInterval( final long[] min, final long[] max )
		{
			super( min, max );
		}

		public ComparableFinalInterval( final Interval interval )
		{
			super( interval );
		}

		@Override
		public boolean equals( final Object other )
		{
			if ( other instanceof Interval )
			{
				final Interval that = ( Interval ) other;
				return Arrays.equals( Intervals.minAsLongArray( that ), min ) && Arrays.equals( Intervals.maxAsLongArray( that ), max );
			}
			return false;
		}

		@Override
		public int hashCode()
		{
			return Arrays.hashCode( min );
		}

		@Override
		public String toString()
		{
			return "(" + Arrays.toString( min ) + " " + Arrays.toString( max ) + ")";
		}

	}

}
