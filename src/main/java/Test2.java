import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Test2
{

	public static void main( final String[] args ) throws IOException
	{
		final String path = "/home/hanslovskyp/local/tmp/sample_A_padded_20160501-argmax.n5/volumes/labels/neuron_ids/label-to-block-mapping/s0/100537";
		final byte[] data = Files.readAllBytes( Paths.get( path ) );
		final ByteBuffer bb = ByteBuffer.wrap( data );
		while ( bb.hasRemaining() )
		{
			System.out.println( bb.getLong() );
		}
	}

}
