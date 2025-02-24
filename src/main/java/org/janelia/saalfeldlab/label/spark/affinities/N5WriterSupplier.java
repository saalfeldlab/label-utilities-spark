package org.janelia.saalfeldlab.label.spark.affinities;

import com.google.gson.GsonBuilder;
import org.janelia.saalfeldlab.label.spark.N5Helpers;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.universe.N5Factory;

import java.io.Serializable;
import java.util.function.Function;
import java.util.function.Supplier;

class N5WriterSupplier implements Supplier<N5Writer>, Serializable {

	private final String container;

	private final boolean withPrettyPrinting;

	private final boolean disableHtmlEscaping;

	private final boolean serializeSpecialFloatingPointValues = true;

	N5WriterSupplier(final String container, final boolean withPrettyPrinting, final boolean disableHtmlEscaping) {

		this.container = container;
		this.withPrettyPrinting = withPrettyPrinting;
		this.disableHtmlEscaping = disableHtmlEscaping;
	}

	@Override
	public N5Writer get() {

		final N5Factory factory = N5Helpers.defaultFactory();
		factory.gsonBuilder(createBuilder());
		return factory.openWriter(container);
	}

	private GsonBuilder createBuilder() {

		return serializeSpecialFloatingPointValues(withPrettyPrinting(disableHtmlEscaping(new GsonBuilder())));
	}

	private GsonBuilder serializeSpecialFloatingPointValues(final GsonBuilder builder) {

		return with(builder, this.serializeSpecialFloatingPointValues, GsonBuilder::serializeSpecialFloatingPointValues);
	}

	private GsonBuilder withPrettyPrinting(final GsonBuilder builder) {

		return with(builder, this.withPrettyPrinting, GsonBuilder::setPrettyPrinting);
	}

	private GsonBuilder disableHtmlEscaping(final GsonBuilder builder) {

		return with(builder, this.disableHtmlEscaping, GsonBuilder::disableHtmlEscaping);
	}

	private static GsonBuilder with(final GsonBuilder builder, boolean applyAction, Function<GsonBuilder, GsonBuilder> action) {

		return applyAction ? action.apply(builder) : builder;
	}
}
