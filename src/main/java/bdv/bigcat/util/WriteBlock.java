package bdv.bigcat.util;

import org.apache.spark.api.java.function.VoidFunction;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.CompressionAdapter;
import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Writer;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class WriteBlock< T, B extends DataBlock< T > > implements VoidFunction< B >
{

	private final String group;

	private final String dataset;

	private final long[] dimensions;

	private final int[] blockSize;

	private final DataType dataType;

	private final String compression;

	public WriteBlock(
			final String group,
			final String dataset,
			final long[] dimensions,
			final int[] blockSize,
			final DataType dataType,
			final String compression )
	{
		super();
		this.group = group;
		this.dataset = dataset;
		this.dimensions = dimensions;
		this.blockSize = blockSize;
		this.dataType = dataType;
		this.compression = compression;
	}

	@Override
	public void call( final B block ) throws Exception
	{
		final N5Writer writer = HDFConverter.n5Writer( group, blockSize );
		final Gson gson = new GsonBuilder()
				.registerTypeHierarchyAdapter( Compression.class, CompressionAdapter.getJsonAdapter() )
				.create();
		writer.writeBlock(
				dataset,
				new DatasetAttributes( dimensions, blockSize, dataType, new GzipCompression() ), // .fromJson(
																									// compression,
																									// Compression.class
																									// )
																									// ),
				block );

	}

}
