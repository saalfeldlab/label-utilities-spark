package bdv.bigcat.util;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.time.DurationFormatUtils;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.CompressionType;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5;
import org.janelia.saalfeldlab.n5.N5Writer;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import bdv.img.cache.VolatileGlobalCellCache;
import bdv.img.h5.H5LabelMultisetSetupImageLoader;
import bdv.labels.labelset.LabelMultisetType;
import bdv.labels.labelset.LabelUtils;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class HDFConverter {

	static public class Parameters
	{
		@Parameter( names = { "--inputfile", "--input", "-i" }, description = "Input HDF5 file")
		public String inputHDF5 = null;
		
		@Parameter( names = { "--groupname", "--group", "-g" }, description = "Output group name (N5 group)")
		public String outputGroupName = null;
		
		@Parameter( names = { "--datasetname", "--data", "-d" }, description = "Output dataset name (N5 relative path from group)")
		public String outputDatasetName = null;
		
		@Parameter( names = { "--cellsize", "-cs"}, description = "Size of cells to use in the output N5 dataset" )
		public List<Integer> cellSize = Arrays.asList(new Integer[] {64, 64, 8});

		@Parameter( names = { "--label", "-l" }, description = "Path to labels in HDF5 file" )
		public String labelPath = "/volumes/labels/neuron_ids";	
		
		@Parameter( names = { "--compression", "-c"}, description = "Compression type to use in output N5 dataset" )
		public String compressionType = "RAW";	
		
		
		public boolean init() {
			if(outputGroupName == null || inputHDF5 == null)
				return false;
			if(outputDatasetName == null)
				outputDatasetName = (inputHDF5.contains(".hdf")?inputHDF5.substring(0, inputHDF5.lastIndexOf(".hdf")):inputHDF5);
			return true;
		}
	}

	public static void main(String[] args) throws IOException
	{
		final Parameters params = new Parameters();
		JCommander jcomm = new JCommander(params, args);
		
		if(!params.init()) {
			jcomm.usage();
			return;
		}
		
		long startTime = System.currentTimeMillis();
		
		int[] cellSize = new int[params.cellSize.size()];
		Arrays.setAll(cellSize, i -> params.cellSize.get(i));
		
		convertHDF5toN5(params.inputHDF5, params.labelPath, cellSize, params.outputGroupName, params.outputDatasetName,
				CompressionType.valueOf(params.compressionType));
		
		long endTime = System.currentTimeMillis();
		
		String formattedTime = DurationFormatUtils.formatDuration(endTime-startTime, "HH:mm:ss.SSS");
		System.out.println("Converted " + params.inputHDF5 + " to N5 dataset at " + params.outputGroupName + " with name " + params.outputDatasetName +
				" in " + formattedTime);
	}
	
	public static void convertHDF5toN5(String hdf5Path, String labelsPath,int[] cellDimensions,
			String outputGroupName, String outputDatasetName, CompressionType compressionType) throws IOException
	{
		IHDF5Reader reader = HDF5Factory.open(hdf5Path);
		
		VolatileGlobalCellCache cache = new VolatileGlobalCellCache(1, 6);
		H5LabelMultisetSetupImageLoader loader = new H5LabelMultisetSetupImageLoader(reader, null, labelsPath, 0, cellDimensions , cache);
		
		RandomAccessibleInterval<LabelMultisetType> img = loader.getImage(0);
		
		int nDim = img.numDimensions();
		
		long[] dimensions = new long[nDim];
		img.dimensions(dimensions);
		
		N5Writer n5 = N5.openFSWriter(outputGroupName);
		n5.createDataset(outputDatasetName, dimensions, cellDimensions, DataType.UINT8, compressionType);
		
		final long[] offset = new long[nDim];
		
		int[] actualCellDimensions = new int[nDim];
		
		for(int d = 0; d < nDim; ) {
			
			Arrays.setAll(actualCellDimensions, i -> (int) Math.min(cellDimensions[i], dimensions[i]-offset[i]));
			
			IntervalView<LabelMultisetType> tm = Views.offsetInterval(img, offset, Arrays.stream(actualCellDimensions).asLongStream().toArray());
						
			final DatasetAttributes attributes = n5.getDatasetAttributes(outputDatasetName);
			
			long[] cellOffset = new long[nDim];
			for(int i = 0; i < nDim; i ++) 
				cellOffset[i] = offset[i]/cellDimensions[i];
			
			final ByteArrayDataBlock dataBlock = new ByteArrayDataBlock(actualCellDimensions, cellOffset,
					LabelUtils.serializeLabelMultisetTypes(Views.flatIterable(tm), (int)Intervals.numElements(actualCellDimensions)));
			
			n5.writeBlock(outputDatasetName, attributes, dataBlock);
	
			for( d = 0; d < nDim; d++) {
				offset[d] += cellDimensions[d];
				if(offset[d] < dimensions[d])
					break;
				else
					offset[d] = 0;
			}
		}
	}
}
