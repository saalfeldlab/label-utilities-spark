package bdv.bigcat.util;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.CompressionType;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import bdv.bigcat.label.FragmentSegmentAssignment;
import bdv.bigcat.ui.GoldenAngleSaturatedARGBStream;
import bdv.bigcat.ui.VolatileLabelMultisetARGBConverter;
import bdv.img.cache.VolatileGlobalCellCache;
import bdv.img.h5.H5LabelMultisetSetupImageLoader;
import bdv.labels.labelset.LabelMultisetType;
import bdv.labels.labelset.LabelUtils;
import bdv.labels.labelset.N5CacheLoader;
import bdv.labels.labelset.VolatileLabelMultisetArray;
import bdv.labels.labelset.VolatileLabelMultisetType;
import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.LocalIdService;
import ch.systemsx.cisd.hdf5.HDF5Factory;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.CacheLoader;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.ref.BoundedSoftRefLoaderCache;
import net.imglib2.cache.util.LoaderCacheAsCacheAdapter;
import net.imglib2.converter.Converters;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.volatiles.VolatileARGBType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class HDFConverter {

	static public class Parameters {
		
		@Parameter( names = { "--inputfile", "--input", "-i" }, description = "Input HDF5 file")
		public String inputHDF5 = null;
		
		@Parameter( names = { "--groupname", "--group", "-g" }, description = "Output group name (N5 group)")
		public String outputGroupName = null;
		
		@Parameter( names = { "--datasetname", "--data", "-d" }, description = "Output dataset name (N5 relative path from group)")
		public String outputDatasetName = null;
		
		@Parameter( names = { "--cellsize", "-c"}, description = "Size of cells to use in the output N5 dataset" )
		public List<Integer> cellSize = Arrays.asList(new Integer[] {64, 64, 8});

		@Parameter( names = { "--label", "-l" }, description = "Path to labels in HDF5 file" )
		public String labelpath = "/volumes/labels/neuron_ids";		
		
		
		public boolean init() {
			if(outputGroupName == null || inputHDF5 == null)
				return false;
			
			if(outputDatasetName == null)
				outputDatasetName = (inputHDF5.contains(".hdf")?inputHDF5.substring(0, inputHDF5.lastIndexOf(".hdf")):inputHDF5);
			
			return true;
		}
	}
	
//	private static int[] cellDimensions = new int[] {64, 64, 8};
//	private static String outputGroupName = "/home/thistlethwaiten/n5-test/";
//	private static String outputDatasetName = "sample_A_n5";

	public static void main(String[] args) throws IOException {
		final Parameters params = new Parameters();
		JCommander jcomm = new JCommander(params, args);
		
		if(!params.init()) {
			jcomm.usage();
			return;
		}
		
		convertHDF5toN5("/home/thistlethwaiten/Downloads/sample_A_20160501.hdf", "/volumes/labels/neuron_ids",
				new int[] {64, 64, 8}, "/home/thistlethwaiten/n5-test/", "sample_A_n5");
		
//		sampleConvert();
//		Bdv bdv = sampleDisplay("sampleA", BdvOptions.options());
//		sampleDisplay("sampleA-downscaled-4x4x1", BdvOptions.options().addTo(bdv));
//		sampleDisplay("sampleA-downscaled-8x8x1", BdvOptions.options().addTo(bdv));
//		sampleDisplay("sampleA-downscaled-16x16x2", BdvOptions.options().addTo(bdv));
//		sampleDisplay("sampleA-downscaled-16x16x4", BdvOptions.options().addTo(bdv));
//		bdv.getBdvHandle().getViewerPanel().setDisplayMode(DisplayMode.SINGLE);
	}
	
	
	public static Bdv sampleDisplay(String gname, String dname, BdvOptions options) throws IOException {

		final N5Reader reader = N5.openFSReader(gname);
		
		final DatasetAttributes attr = reader.getDatasetAttributes(dname);
		
		long[] dim = attr.getDimensions();
		int[] blocksize = attr.getBlockSize();
		
		final CacheLoader< Long, Cell< VolatileLabelMultisetArray > > cacheLoader = new N5CacheLoader(reader, dname);
		
		final BoundedSoftRefLoaderCache< Long, Cell< VolatileLabelMultisetArray > > cache = new BoundedSoftRefLoaderCache<>( 100 );
		final LoaderCacheAsCacheAdapter< Long, Cell< VolatileLabelMultisetArray > > wrappedCache = new LoaderCacheAsCacheAdapter<>( cache, cacheLoader );
		
		final CachedCellImg<VolatileLabelMultisetType,VolatileLabelMultisetArray> img = new CachedCellImg<VolatileLabelMultisetType,VolatileLabelMultisetArray>(
				new CellGrid(dim, blocksize), new VolatileLabelMultisetType(), wrappedCache, new VolatileLabelMultisetArray(0, true));
		
		final GoldenAngleSaturatedARGBStream argbStream = new GoldenAngleSaturatedARGBStream( new FragmentSegmentAssignment( new LocalIdService() ) );
		final RandomAccessibleInterval<VolatileARGBType> volatileConverted = Converters.convert((RandomAccessibleInterval<VolatileLabelMultisetType>)img, new VolatileLabelMultisetARGBConverter( argbStream ), new VolatileARGBType());

		return BdvFunctions.show(volatileConverted, "BDV - " + dname, options);
	}
	
	
	public static void convertHDF5toN5(String hdf5Path, String labelsPath, int[] cellDimensions, String outputGroupName, String outputDatasetName) throws IOException {
		
		IHDF5Reader reader = HDF5Factory.open("/home/thistlethwaiten/Downloads/sample_A_20160501.hdf");
		
		VolatileGlobalCellCache cache = new VolatileGlobalCellCache(1, 6);
		H5LabelMultisetSetupImageLoader loader = new H5LabelMultisetSetupImageLoader(reader, null, "/volumes/labels/neuron_ids", 0, cellDimensions , cache);
		
		RandomAccessibleInterval<LabelMultisetType> img = loader.getImage(0);
		
		int nDim = img.numDimensions();
		
		System.out.println(String.format("Has %d dimensions", nDim));
		
		long[] dimensions = new long[nDim];
		img.dimensions(dimensions);
		
		System.out.println(String.format("Dims are %s", Arrays.toString(dimensions)));

		
		N5Writer n5 = N5.openFSWriter(outputGroupName);
		n5.createDataset(outputDatasetName, dimensions, cellDimensions, DataType.UINT8, CompressionType.RAW);
		
		
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
				offset[d] += cellDimensions[d]; // parallelSize is in units of CELLS
				// (offset is in pixels)
				if(offset[d] < dimensions[d])
					break;
				else
					offset[d] = 0;
			}
		}
		
		System.out.println("done");
	}
}
