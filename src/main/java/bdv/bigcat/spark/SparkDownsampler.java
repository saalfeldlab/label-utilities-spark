package bdv.bigcat.spark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.CompressionType;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

public class SparkDownsampler
{
	private static String groupName = "/home/thistlethwaiten/n5-test";
	private static String datasetName = "sample-fullres-b3x3";
	private static String downsampledDatasetName = "testspark6-downscaled-13x13-b3x3";

	private DatasetAttributes attributes;
	private long[] dimensions;
	private int[] blockSize;

	// TODO change this to either take the attributes, a reader and a datasetname, or a groupname and a datasetname,
	// and do the rest automatically
	public SparkDownsampler(N5Reader reader, String readDatasetName) throws IOException {
		this.attributes = reader.getDatasetAttributes(readDatasetName);
		this.dimensions = attributes.getDimensions();
		this.blockSize = attributes.getBlockSize();
	}
	
	public static void main(String[] args) throws IOException {
		SparkConf conf      = new SparkConf().setMaster("local").setAppName( "SparkDownsampler" );
		JavaSparkContext sc = new JavaSparkContext( conf );
		new SparkDownsampler(new N5FSReader(groupName), datasetName).downsample(
				sc, new int[] {13,13}, new int[] {3,8}, groupName, downsampledDatasetName, CompressionType.RAW);
	}
	
	// TODO test with a non-1 block size
	// parallelSize is (arbitrarily) in CELLS/blocks in the downscaled space
	// i.e. if you specify parallelSize = 3x3 when cellsize is 2x2 and the downscaling is 5x5,
	// each spark parallel thing will be dealing with 9 cells, each of which represents 4 * 25 = 100 pixels of full-res space,
	// = 900 pixels per spark parallelization
	
	
	// in spark it's going to read in from N5 using the cell coordinates...
	public void downsample(JavaSparkContext sc, int[] downsampleFactor, int[] parallelSize, String outputGroupName, String outputDatasetName, CompressionType compressionType) throws IOException {
		List<DownsampleBlock> parallelizeSections = new ArrayList<DownsampleBlock>();

		int nDim = attributes.getNumDimensions();
		
		long[] downsampledDimensions = new long[nDim];
		Arrays.setAll(downsampledDimensions, i -> (int)(Math.ceil((double)dimensions[i]/downsampleFactor[i])));

		final long[] offset = new long[nDim];
		int[] actualSize = new int[nDim];
		
		for(int d = 0; d < nDim; ) {
			
			for(int i = 0; i < nDim; i ++) {
				actualSize[i] = (int) Math.min(parallelSize[i]*blockSize[i], downsampledDimensions[i]-offset[i]);
			}
			// actualSize is in pixels, not sure why (?), after parallelization it gets translated back to cells or something

//			actualSize[i] = (int) ((offset[i]+(parallelSize[i]*blockSize[i]) < dimensions[i]) ? parallelSize[i]*blockSize[i] : dimensions[i]-offset[i]);
			
			
//			System.out.println("===Offset is " + Arrays.toString(offset));
			parallelizeSections.add(new DownsampleBlock(offset.clone(), actualSize.clone()));
			
			
			for( d = 0; d < nDim; d++) {
				offset[d] += parallelSize[d]*blockSize[d];
				if(offset[d] < downsampledDimensions[d])
					break;
				else
					offset[d] = 0;
			}
		}

		N5Writer writer = new N5FSWriter(outputGroupName);
		writer.createDataset(outputDatasetName, downsampledDimensions, blockSize, DataType.UINT8, compressionType);
		
		downsampleHelper(sc, parallelizeSections, downsampleFactor, outputGroupName, outputDatasetName);
	}
	
	private static void downsampleHelper(JavaSparkContext sc, List<DownsampleBlock> parallelizeSections, int[] downsampleFactor, String outputGroupName, String outputDatasetName ) {
		
		Integer output = sc.parallelize(parallelizeSections)
			.map( new SparkDownsampleFunction(groupName, datasetName, downsampleFactor, outputGroupName, outputDatasetName))
			.reduce( (i,j) -> i+j);
		
		System.out.println("output is " + output);
	}
}