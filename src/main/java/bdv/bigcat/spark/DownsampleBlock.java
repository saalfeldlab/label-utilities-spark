package bdv.bigcat.spark;

import java.io.Serializable;
import java.util.Arrays;

public class DownsampleBlock implements Serializable
{
	private static final long serialVersionUID = -1784788228602767119L;
	private final long[] min;
	private final int[] size;

	public DownsampleBlock(long[] min, int[] size) {
		this.min = min;
		this.size = size;
	}
	
	public long[] getMin() { return min; }
	public int[] getSize() { return size; }
	
	@Override
	public String toString() { return "DownsampleBlock[min=" + Arrays.toString(min) + ",size=" + Arrays.toString(size) + "]"; }
}