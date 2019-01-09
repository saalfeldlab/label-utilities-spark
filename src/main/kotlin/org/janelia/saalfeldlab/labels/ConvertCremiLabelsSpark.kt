package org.janelia.saalfeldlab.labels

import net.imglib2.RandomAccessibleInterval
import net.imglib2.img.array.ArrayImgFactory
import net.imglib2.loops.LoopBuilder
import net.imglib2.type.NativeType
import net.imglib2.type.numeric.integer.UnsignedLongType
import net.imglib2.type.numeric.real.DoubleType
import net.imglib2.util.Intervals
import net.imglib2.util.StopWatch
import net.imglib2.util.Util
import net.imglib2.view.Views
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.function.Function2
import org.janelia.saalfeldlab.n5.DataType
import org.janelia.saalfeldlab.n5.DatasetAttributes
import org.janelia.saalfeldlab.n5.GzipCompression
import org.janelia.saalfeldlab.n5.N5FSWriter
import org.janelia.saalfeldlab.n5.hdf5.N5HDF5Reader
import org.janelia.saalfeldlab.n5.imglib2.N5Utils
import org.slf4j.LoggerFactory
import picocli.CommandLine
import scala.Tuple2
import java.lang.invoke.MethodHandles
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.function.BiConsumer
import kotlin.math.max
import kotlin.math.min

private val USER_HOME = System.getProperty("user.home")

class ConvertCremiLabelsSpark
{
	class CmdlineArgs {
		@CommandLine.Parameters(index= "0", arity = "1", paramLabel = "INPUT_CONTAINER", description = arrayOf("HDF5"))
		var inputContainer: String? = null

		@CommandLine.Parameters(index = "1", arity = "1", paramLabel = "OUTPUT_CONTAINER", description = arrayOf("N5"))
		var outputContainer: String? = null

		@CommandLine.Option(names = arrayOf("--input-dataset", "-i"), paramLabel = "INPUT_DATASET", defaultValue = "volumes/labels/neuron_ids")
		var inputDataset: String = "volumes/labels/neuron_ids"

		@CommandLine.Option(names = arrayOf("--output-dataset", "-o"), paramLabel = "OUTPUT_DATASET", defaultValue = "\${INPUT_DATASET}")
		var outputDataset: String? = null

		@CommandLine.Option(names = arrayOf("--num-fillers", "-n"), paramLabel = "NUM_FILLERS", defaultValue = "2", showDefaultValue = CommandLine.Help.Visibility.ON_DEMAND)
		var numFillers: Int = 2

		@CommandLine.Option(names = arrayOf("--block-size"), paramLabel = "BLOCK_SIZE", defaultValue = "64", split = ",")
		var blockSize: IntArray = intArrayOf(64, 64, 64)

		@CommandLine.Option(names = arrayOf("--z-min"), defaultValue = "${Long.MIN_VALUE}")
		var zMin: Long = Long.MIN_VALUE

		@CommandLine.Option(names = arrayOf("--z-max"), defaultValue = "${Long.MAX_VALUE}")
		var zMax: Long = Long.MAX_VALUE
	}

	companion object {
	    val LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass())
	}
}

private fun <T: NativeType<T>> asArrayImg(input: RandomAccessibleInterval<T>): RandomAccessibleInterval<T> {
	val output = ArrayImgFactory(Util.getTypeFromInterval(input)).create(input)
	LoopBuilder.setImages(input, output).forEachPixel(BiConsumer { i, o ->  o.set(i)})
	return output
}


private fun runSpark(
		sc: JavaSparkContext,
		inputContainer: String,
		outputContainer: String,
		inputDataset: String,
		outputDataset: String,
		imgDim: LongArray,
		zMin: Long,
		zMax: Long,
		numFillers: Long,
		blockSize: IntArray) {
	val blockSizeZ = blockSize[2]
	sc.parallelize((zMin until zMax).toList())
			.mapToPair {
				val im = N5Utils.open<UnsignedLongType>(N5HDF5Reader(inputContainer, imgDim[0].toInt(), imgDim[1].toInt(), 1), inputDataset)
				Tuple2(it, Tuple2(asArrayImg(Views.hyperSlice(im, 2, it)), asArrayImg(Views.hyperSlice(im, 2, it+1)))) }
			.mapToPair {
				val fillers = InterpolateBetweenSections.makeFillers(numFillers, imgDim[0], imgDim[1])
				InterpolateBetweenSections.interpolateBetweenSectionsWithSignedDistanceTransform(it._2()._1(), it._2()._2(), ArrayImgFactory(DoubleType()), *fillers)
				Tuple2(it._1(), listOf(it._2()._1()) + fillers + (if (it._1() == zMax - 1) listOf(it._2()._2()) else listOf()))
			}
			.map { it._2().mapIndexed { index, rai -> Tuple2(index + numFillers * (it._1() - zMin), rai) }}
			.flatMapToPair { it.iterator() }
			.mapToPair {Tuple2(it._1() / blockSizeZ, it)}
			.aggregateByKey(mutableListOf<Tuple2<Long, RandomAccessibleInterval<UnsignedLongType>>>(), { l, t -> l?.add(t); l!! }, { l1, l2 -> (l1!! + l2!!).toMutableList() } )
			.mapToPair {Tuple2(it._1() * blockSizeZ, it._2()) }
			.mapValues { it.sortedBy { it._1() } }
			.mapValues { Views.stack(it.map { it._2() }) }
			.map { Views.translate(it._2(), 0, 0, it._1())}
			.foreach { N5Utils.save(it as RandomAccessibleInterval<UnsignedLongType>, N5FSWriter(outputContainer), outputDataset, blockSize, GzipCompression()) }
}

fun main(mainArgs: Array<String>) {

	val args = ConvertCremiLabelsSpark.CmdlineArgs()
	CommandLine.populateCommand(args, *mainArgs)

	val inputContainer = args.inputContainer!!
	val outputContainer = args.outputContainer!!
	val inputDataset = args.inputDataset
	val outputDataset = args.outputDataset?: inputDataset
	val numFillers = max(args.numFillers.toLong(), 0L)
	val reader = N5HDF5Reader(inputContainer, 1250, 1250, 1)
	val blockSize: IntArray = if (args.blockSize.size == 3) args.blockSize else IntArray(3, {args.blockSize[0]})
	val img = N5Utils.open<UnsignedLongType>(reader, inputDataset)
	val imgDim = Intervals.dimensionsAsLongArray(img)
	ConvertCremiLabelsSpark.LOG.info("imgDim={}", imgDim)

	val zMin = max(args.zMin, img.min(2))
	val zMax = min(args.zMax, img.max(2))

	val stopWatch = StopWatch()
	val conf = SparkConf()
			.setAppName("Upsample cremi labels")
			.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

	val targetDims = longArrayOf(img.dimension(0), img.dimension(1), (zMin - zMax) * (1 + numFillers) + 1)
	N5FSWriter(outputContainer).createDataset(outputDataset, targetDims, blockSize, DataType.UINT64, GzipCompression())

	JavaSparkContext(conf).use {
		stopWatch.start()
		runSpark(it, inputContainer, outputContainer, inputDataset, outputDataset, imgDim, zMin, zMax, numFillers, blockSize)
		stopWatch.stop()
	}

	println("Upsampled ${zMax - zMin + 1} sections with an additional ${numFillers} fillers between each pair of original sections in ${TimeUnit.SECONDS.convert(stopWatch.nanoTime(), TimeUnit.NANOSECONDS)} seconds")

}
