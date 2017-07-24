package bdv.bigcat.spark;

import java.util.Arrays;

import org.apache.spark.api.java.function.Function;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import bdv.labels.labelset.LabelMultisetType;
import bdv.labels.labelset.LabelMultisetTypeDownscaler;
import bdv.labels.labelset.N5CacheLoader;
import bdv.labels.labelset.VolatileLabelMultisetArray;
import net.imglib2.cache.CacheLoader;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.ref.BoundedSoftRefLoaderCache;
import net.imglib2.cache.util.LoaderCacheAsCacheAdapter;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.view.Views;

public class SparkDownsampleFunction implements Function<DownsampleBlock, Integer> {
	
	private static final long serialVersionUID = 1384028449836651388L;
	private final String groupName;
	private final String datasetName;
	private final int[] factor;
	private final String outputGroupName;
	private final String outputDatasetName;

	public SparkDownsampleFunction(String groupName, String datasetName, int[] factor, String outputGroupName, String outputDatasetName ) {
		this.groupName = groupName;
		this.datasetName = datasetName;
		this.factor = factor;
		this.outputGroupName = outputGroupName;
		this.outputDatasetName = outputDatasetName;
	}

	@Override
	public Integer call(DownsampleBlock targetRegion) throws Exception {
		
//		System.out.println("SparkDownsampleFunction.call() with targetRegion of " + targetRegion);
		
		N5Reader reader = new N5FSReader(groupName);
		DatasetAttributes attr = reader.getDatasetAttributes(datasetName);
		
		long[] dimensions = attr.getDimensions();
		int[] blocksize = attr.getBlockSize();
		
		int nDim = dimensions.length;
		final long[] offset = new long[nDim];
		
		int[] targetSize = targetRegion.getSize();
		long[] targetMin = targetRegion.getMin();
		
		long[] actualLocation = new long[nDim];
		long[] actualSize = new long[nDim];
		
		
		final CacheLoader< Long, Cell< VolatileLabelMultisetArray > > cacheLoader = new N5CacheLoader(reader, datasetName);
		
		// TODO 10 is sort of arbitrary for maxSoftRefs
		final BoundedSoftRefLoaderCache< Long, Cell< VolatileLabelMultisetArray > > cache = new BoundedSoftRefLoaderCache<>( 10 );
		final LoaderCacheAsCacheAdapter< Long, Cell< VolatileLabelMultisetArray > > wrappedCache = new LoaderCacheAsCacheAdapter<>( cache, cacheLoader );
		
		final CachedCellImg<LabelMultisetType,VolatileLabelMultisetArray> inputImg = new CachedCellImg<LabelMultisetType,VolatileLabelMultisetArray>(
				new CellGrid(dimensions, blocksize), new LabelMultisetType(), wrappedCache, new VolatileLabelMultisetArray(0, true));
		
		VolatileLabelMultisetArray downscaledCell;
		
		int numCellsDownscaled = 0;
		
		final N5Writer writer = new N5FSWriter(outputGroupName);
		final DatasetAttributes writerAttributes = writer.getDatasetAttributes(outputDatasetName);
		
		long[] writeLocation = new long[nDim];
		
		for(int d = 0; d < nDim; ) {
			
			// this needs to be redone for the bounds checking TODO
			Arrays.setAll(actualLocation, i -> factor[i] * (targetMin[i] + offset[i]));
			Arrays.setAll(actualSize, i -> Math.min(
					factor[i] * (offset[i] + blocksize[i] > targetMin[i] + targetSize[i] ? (targetMin[i] + targetSize[i] - offset[i]) : blocksize[i]),
					dimensions[i] - actualLocation[i])
					);
			
//			System.out.println("  telling downscaler to create cell from actualLocation="+Arrays.toString(actualLocation)+",actualSize="+Arrays.toString(actualSize));
			
			downscaledCell = LabelMultisetTypeDownscaler.createDownscaledCell(Views.offsetInterval(inputImg, actualLocation, actualSize), factor);
			
			byte[] bytes = new byte[LabelMultisetTypeDownscaler.getSerializedVolatileLabelMultisetArraySize(downscaledCell)];
			LabelMultisetTypeDownscaler.serializeVolatileLabelMultisetArray(downscaledCell, bytes);
			
			// TODO check: is the block size here already capped at edge cases (I don't think it is), does it need to be? etc.
//			System.out.println("I am a spark downsampler, offset=" + Arrays.toString(offset));
			
			for(int i = 0; i < nDim; i ++) writeLocation[i] = (targetMin[i] + offset[i])/blocksize[i];
			
			final ByteArrayDataBlock dataBlock = new ByteArrayDataBlock(blocksize, writeLocation, bytes);
			writer.writeBlock(outputDatasetName, writerAttributes, dataBlock);
			
			
			numCellsDownscaled++;
			
			
			for( d = 0; d < nDim; d++) {
				offset[d] += blocksize[d];
				if(offset[d] < targetSize[d])
					break;
				else
					offset[d] = 0;
			}
		}
		
		return numCellsDownscaled;
	}

}
