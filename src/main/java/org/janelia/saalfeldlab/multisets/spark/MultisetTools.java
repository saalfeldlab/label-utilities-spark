package org.janelia.saalfeldlab.multisets.spark;

import java.util.Arrays;
import java.util.Optional;
import java.util.concurrent.Callable;

import org.janelia.saalfeldlab.multisets.spark.MultisetTools.Tool.FromString;
import org.janelia.saalfeldlab.multisets.spark.convert.ConvertToLabelMultisetType;
import org.janelia.saalfeldlab.multisets.spark.downsample.SparkDownsampler;
import org.janelia.saalfeldlab.multisets.spark.uniquelabels.ExtractUniqueLabelsPerBlock;
import org.janelia.saalfeldlab.multisets.spark.uniquelabels.downsample.LabelListDownsampler;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.ITypeConverter;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

public class MultisetTools
{

	public static enum Tool
	{
		CONVERT( ConvertToLabelMultisetType::run ),
		DOWNSAMPLE( SparkDownsampler::run ),
		EXTRACT_UNIQUE_LABELS( ExtractUniqueLabelsPerBlock::run ),
		DOWNSAMPLE_UNIQUE_LABELS( LabelListDownsampler::run );

		private interface ExceptionConsumer< T >
		{
			public void accept( T t ) throws Exception;
		}

		private final ExceptionConsumer< String[] > run;

		private Tool( final ExceptionConsumer< String[] > run )
		{
			this.run = run;
		}

		public String getCmdLineRepresentation()
		{
			return this.name().toLowerCase();
		}

		public static Tool fromCmdLineRepresentation( final String representation )
		{
			return Tool.valueOf( representation.replace( "-", "_" ).toUpperCase() );
		}

		public static class FromString implements ITypeConverter< Tool >
		{

			@Override
			public Tool convert( final String str ) throws Exception
			{
				return Tool.fromCmdLineRepresentation( str );
			}

		}

	}

	@Command( name = "multiset-tools" )
	public static class CommandLineParameters implements Callable< Boolean >
	{

		@Parameters(
				index = "0",
				paramLabel = "TOOL",
				converter = FromString.class,
				description = "Tool to run. Run multiset-tools <TOOL> --help/-h for specific help message. Current options are convert, downsample" )
		private Tool tool;

		@Option( names = { "-h", "--help" }, usageHelp = true, description = "display a help message" )
		private boolean helpRequested;

		@Override
		public Boolean call() throws Exception
		{
			return true;
		}

	}

	public static void main( final String[] args ) throws Exception
	{
		final CommandLineParameters params = new CommandLineParameters();
		final Boolean paramsParsedSuccessfully = Optional.ofNullable( CommandLine.call( params, System.err, args.length > 0 ? args[ 0 ] : "--help" ) ).orElse( false );
		if ( paramsParsedSuccessfully )
		{
			params.tool.run.accept( Arrays.copyOfRange( args, 1, args.length ) );
		}
	}

}
