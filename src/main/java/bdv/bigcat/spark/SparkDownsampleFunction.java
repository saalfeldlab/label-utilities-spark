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
import bdv.labels.labelset.LabelUtils;
import bdv.labels.labelset.N5CacheLoader;
import bdv.labels.labelset.VolatileLabelMultisetArray;
import net.imglib2.RandomAccessible;
import net.imglib2.cache.CacheLoader;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.ref.BoundedSoftRefLoaderCache;
import net.imglib2.cache.util.LoaderCacheAsCacheAdapter;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.view.Views;

public class SparkDownsampleFunction implements Function<DownsampleBlock, Integer> {
	
	private static final long serialVersionUID = 1384028449836651388L;
	private final String inputGroupName;
	private final String inputDatasetName;
	private final int[] factor;
	private final String outputGroupName;
	private final String outputDatasetName;

	public SparkDownsampleFunction(String inputGroupName, String inputDatasetName, int[] factor, String outputGroupName, String outputDatasetName ) {
		this.inputGroupName = inputGroupName;
		this.inputDatasetName = inputDatasetName;
		this.factor = factor;
		this.outputGroupName = outputGroupName;
		this.outputDatasetName = outputDatasetName;
	}

	@Override
	public Integer call(DownsampleBlock targetRegion) throws Exception {
		
		System.out.println("Called with downsamp block " + targetRegion);

		N5Reader reader = new N5FSReader(inputGroupName);
		DatasetAttributes attr = reader.getDatasetAttributes(inputDatasetName);
		
		long[] dimensions = attr.getDimensions();
		int[] blocksize = attr.getBlockSize();
		
		int nDim = dimensions.length;
		final long[] offset = new long[nDim];
		
		int[] targetSize = targetRegion.getSize();
		long[] targetMin = targetRegion.getMin();
		
		long[] actualLocation = new long[nDim];
		long[] actualSize = new long[nDim];
		
		
		final CacheLoader< Long, Cell< VolatileLabelMultisetArray > > cacheLoader = new N5CacheLoader(reader, inputDatasetName);
		
		final BoundedSoftRefLoaderCache< Long, Cell< VolatileLabelMultisetArray > > cache = new BoundedSoftRefLoaderCache<>( 1 );
		final LoaderCacheAsCacheAdapter< Long, Cell< VolatileLabelMultisetArray > > wrappedCache = new LoaderCacheAsCacheAdapter<>( cache, cacheLoader );
		
		final CachedCellImg<LabelMultisetType,VolatileLabelMultisetArray> inputImg = new CachedCellImg<LabelMultisetType,VolatileLabelMultisetArray>(
				new CellGrid(dimensions, blocksize), new LabelMultisetType(), wrappedCache, new VolatileLabelMultisetArray(0, true));
		
		final RandomAccessible<LabelMultisetType> extendedImg = Views.extendValue(inputImg, LabelUtils.getOutOfBounds());
		
		VolatileLabelMultisetArray downscaledCell;
		
		int numCellsDownscaled = 0;
		
		final N5Writer writer = new N5FSWriter(outputGroupName);
		final DatasetAttributes writerAttributes = writer.getDatasetAttributes(outputDatasetName);
		
		long[] writeLocation = new long[nDim];
		
		for(int d = 0; d < nDim; ) {
			
			Arrays.setAll(actualLocation, i -> factor[i] * (targetMin[i] + offset[i]));
			Arrays.setAll(actualSize, i -> factor[i] * blocksize[i]);
			
//			Arrays.setAll(actualSize, i -> Math.min(
//					factor[i] * (offset[i] + blocksize[i] > targetMin[i] + targetSize[i] ? (targetMin[i] + targetSize[i] - offset[i]) : blocksize[i]),
//					dimensions[i] - actualLocation[i]));

//			downscaledCell = LabelMultisetTypeDownscaler.createDownscaledCell(Views.offsetInterval(inputImg, actualLocation, actualSize), factor);
			downscaledCell = LabelMultisetTypeDownscaler.createDownscaledCell(Views.offsetInterval(extendedImg, actualLocation, actualSize), factor);
			
			byte[] bytes = new byte[LabelMultisetTypeDownscaler.getSerializedVolatileLabelMultisetArraySize(downscaledCell)];
			LabelMultisetTypeDownscaler.serializeVolatileLabelMultisetArray(downscaledCell, bytes);
			
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
