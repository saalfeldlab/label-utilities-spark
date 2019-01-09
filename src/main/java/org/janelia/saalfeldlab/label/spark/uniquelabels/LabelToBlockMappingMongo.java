package org.janelia.saalfeldlab.label.spark.uniquelabels;

import org.apache.commons.lang3.time.DurationFormatUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.label.spark.exception.InvalidDataset;
import org.janelia.saalfeldlab.label.spark.exception.InvalidN5Container;
import org.janelia.saalfeldlab.labels.blocks.mongo.LabelBlockLookupFromMongo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Parameters;

import java.io.IOException;
import java.io.Serializable;
import java.lang.invoke.MethodHandles;
import java.util.concurrent.Callable;
import java.util.function.Supplier;

public class LabelToBlockMappingMongo
{

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static class CommandLineParameters implements Callable< Void >
	{

		@Parameters( index = "0", paramLabel = "INPUT_N5", description = "Input N5 container (has to be file system)" )
		private String inputN5;

		@Parameters( index = "1", paramLabel = "INPUT_DATASET", description = "Input dataset. Must be LabelMultisetType or integer type" )
		private String inputDataset;

		@CommandLine.Option(names = "{--hostname, -h}", paramLabel = "HOSTNAME", description = "Mongo database hostname or ip address. Defaults to localhost")
		private String hostname = "localhost";

		@CommandLine.Option(names = {"--port", "-p"}, paramLabel = "PORT", description = "Mongo database port. Deafults to 27017")
		private int port = 27017;

		@CommandLine.Option(names = {"--database-name", "-d"}, paramLabel = "DATABASE_NAME", description = "Name of database that holds label-block-lookup collections for each scale level. Defaults to INPUT_DATASET-lookup")
		private String databaseName;

		@CommandLine.Option(names = {"--collection-pattern", "-c"}, paramLabel = "COLLECTION_PATTERN", description = "Pattern for collections for label-block-lookup at each level. Must contain exactly one `%d'. Defaults to `%d'")
		private String collectionPattern = "%d";

		@Override
		public Void call() throws Exception
		{
			if ( this.inputN5 == null ) { throw new InvalidN5Container( "INPUT_N5", this.inputN5 ); }

			if ( this.inputDataset == null ) { throw new InvalidDataset( "INPUT_DATASET", inputDataset ); }

			this.hostname = this.hostname == null ? this.inputN5 : this.hostname;

			this.databaseName = databaseName == null ? this.inputDataset.replace("/","_").replace("\\", "_") + "-lookup" : this.databaseName;

			final long startTime = System.currentTimeMillis();

			final SparkConf conf = new SparkConf().setAppName( MethodHandles.lookup().lookupClass().getName() );

			try (final JavaSparkContext sc = new JavaSparkContext( conf ))
			{
//				@LabelBlockLookup.Parameter private val hostname: String = LOCALHOST,
//				@LabelBlockLookup.Parameter private val port: Int = DEFAULT_PORT,
//				@LabelBlockLookup.Parameter private val databaseName: String = DEFAULT_DATABASE_NAME,
//				@LabelBlockLookup.Parameter private val collectionPattern: String = DEFAULT_COLLECTION_PATTERN,
//				@LabelBlockLookup.Parameter private val idKey: String = IntervalArrayStoreDocumentDocumentCodec.DEFAULT_ID_KEY,
//				@LabelBlockLookup.Parameter private val intervalsKey: String = IntervalArrayStoreDocumentDocumentCodec.DEFAULT_INTERVALS_KEY)
				final Supplier<LabelBlockLookupFromMongo> lookup = new LabelBlockLookupFromMongoSupplier(
						hostname,
						port,
						databaseName,
						collectionPattern,
						"_id",
						"intervals"
				);

//						() -> new LabelBlockLookupFromMongo(
//						hostname,
//						port,
//						databaseName,
//						collectionPattern,
//						"_id",
//						"intervals");

				LabelToBlockMapping.createMappingWithMultiscaleCheck(sc, inputN5, inputDataset, lookup);
			}

			final long endTime = System.currentTimeMillis();

			final String formattedTime = DurationFormatUtils.formatDuration( endTime - startTime, "HH:mm:ss.SSS" );
			LOG.info( "Created label to block mapping for " + inputN5 + ":" + inputDataset + " at " + port + " in " + formattedTime );
			return null;

		}
	}

	public static void run( final String[] args ) throws IOException
	{
		CommandLine.call( new CommandLineParameters(), System.err, args );
	}

	public static void main( final String[] args ) throws IOException
	{
		run( args );
	}

	private static class LabelBlockLookupFromMongoSupplier implements Supplier<LabelBlockLookupFromMongo>, Serializable
	{

		private final String hostname;

		private final int port;

		private final String databaseName;

		private final String collectionPattern;

		private final String idKey;

		private final String intervalsKey;

		private LabelBlockLookupFromMongoSupplier(
				String hostname,
				int port,
				String databaseName,
				String collectionPattern,
				String idKey,
				String intervalsKey) {
			this.hostname = hostname;
			this.port = port;
			this.databaseName = databaseName;
			this.collectionPattern = collectionPattern;
			this.idKey = idKey;
			this.intervalsKey = intervalsKey;
		}

		@Override
		public LabelBlockLookupFromMongo get() {
			return new LabelBlockLookupFromMongo(
					hostname,
					port,
					databaseName,
					collectionPattern,
					"_id",
					"intervals");
		}
	}


}
