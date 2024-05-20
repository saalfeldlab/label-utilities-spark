package org.janelia.saalfeldlab.label.spark.affinities;

import com.google.gson.annotations.Expose;
import org.apache.commons.lang3.builder.ToStringBuilder;
import picocli.CommandLine;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.Stream;

public class Offset implements Serializable {

	@Expose
	private final int channelIndex;

	@Expose
	private final long[] offset;

	public Offset(final int channelIndex, final long... offset) {

		this.channelIndex = channelIndex;
		this.offset = offset;
	}

	public long[] offset() {

		return offset.clone();
	}

	public int channelIndex() {

		return channelIndex;
	}

	@Override
	public String toString() {

		return new ToStringBuilder(this)
				.append("channelIndex", channelIndex)
				.append("offset", Arrays.toString(offset))
				.toString();
	}

	public static Offset parseOffset(final String representation) {

		final String[] split = representation.split(":");
		return new Offset(
				split.length > 1 ? Integer.parseInt(split[1]) : -1,
				Stream.of(split[0].split(",")).mapToLong(Long::parseLong).toArray());
	}

	public static class Converter implements CommandLine.ITypeConverter<Offset> {

		@Override
		public Offset convert(String s) {

			return Offset.parseOffset(s);
		}
	}
}
