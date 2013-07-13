package com.github.hajile.symbola.fn

import com.github.hajile.symbola.cl.{MatrixFloatBuffer, OpenCLExample}
import com.google.common.base.Charsets
import com.google.common.io.Resources
import com.jogamp.opencl.CLMemory.Mem
import com.jogamp.opencl.{CLEventList, CLEvent, CLBuffer, CLContext}
import java.nio.FloatBuffer
import scala.collection.mutable

class ExprGraph(val clContext: CLContext) {

  import ExprGraph._

  private val queue = clContext.getMaxFlopsDevice.createCommandQueue()
  private val program = clContext.createProgram(Resources.toString(classOf[OpenCLExample].getResource("kernels/kernels.cl"), Charsets.UTF_8))
  //  program.setFastRelaxedMath()
  //  program.setUnsafeMathOptimizations()
  program.build()

  private val variables = mutable.HashMap[String, M.InputCell]()
  private val roots = mutable.HashMap[String, M.OutputCell]()

  private val inputValues = mutable.HashMap[String, BufferWithShape]()
  //  private val outputValues = mutable.HashMap[String, BufferWithShape]()

  private var realized: Option[Map[MatrixExpr, RealizedExpr]] = None

  def in(name: String): M.InputCell = {
    require(!variables.contains(name))

    val c = M.InputCell(name)
    variables += (name -> c)
    c
  }

  def out(name: String, expr: MatrixExpr): M.OutputCell = {
    require(!roots.contains(name))

    val c = M.OutputCell(name, expr)
    roots += (name -> c)
    c
  }

  def inValue(name: String): BufferWithShape = inputValues(name)

  def getIn(name: String, shape: RealizedMatrix): MatrixFloatBuffer = {
    val clBuf = alloc(shape.sizeP)(variables(name))
    inputValues += (name -> BufferWithShape(clBuf, shape))
    new MatrixFloatBuffer(clBuf.getBuffer, shape)
  }

  def writeIn(name: String) {
    queue.putWriteBuffer(inputValues(name).buf, true)
  }

  def alloc(size: Int)(implicit e: MatrixExpr): FloatCLBuffer = {
    println(f"Allocating CL Buffer with size ${size * 4 / 1048576.0}%.2fMiB for expression $e")
    clContext.createFloatBuffer(size, Mem.READ_WRITE)
  }

  def realize() {
    require(variables.keySet == inputValues.keySet, "Some inputs aren't initialized")

    val cache = mutable.HashMap[MatrixExpr, RealizedExpr]()

    def realizeExpr(e: MatrixExpr): RealizedExpr = {
      def r(implicit e: MatrixExpr, shape: RealizedShape): RealizedExpr = {
        require(!cache.contains(e))
        val ret = RealizedExpr(e, BufferWithShape(alloc(shape.sizeP), shape))
        cache += e -> ret
        ret
      }

      if (cache.contains(e))
        cache(e)
      else e match {
        case c@M.InputCell(name) =>
          require(variables.contains(name), "Where did you get that variable? It doesn't belong to this ExprGraph")
          val ret = RealizedExpr(c, inputValues(name))
          cache += e -> ret
          ret
        //          r(c, inputValues(name).shape)
        //        case e@M.Cos(i) =>
        //          r(e, realizeExpr(i).out.shape)
        case e@M.Dotwise1(i, _, _) =>
          r(e, realizeExpr(i).out.shape)
        case e@M.OutputCell(_, i) =>
          val ret = RealizedExpr(e, realizeExpr(i).out)
          cache += e -> ret
          ret
        case e@M.Prod(i1, i2) =>
          val shape1 = realizeExpr(i1).out.shape.asInstanceOf[RealizedMatrix]
          val shape2 = realizeExpr(i2).out.shape.asInstanceOf[RealizedMatrix]
          require(shape1.cols == shape2.rows)
          r(e, RealizedMatrix(shape1.rows, shape2.cols))
        //        case e@Sin(i) =>
        //          r(e, realizeExpr(i).out.shape)
        case e@M.Sum(is) =>
          val shapes = is.map(realizeExpr(_).out.shape)
          require(shapes.forall(_ == shapes(0)))
          r(e, shapes(0))
        case e@M.Transpose(i) =>
          r(e, realizeExpr(i).out.shape)
      }
    }

    for (r <- roots.values) {
      realizeExpr(r)
    }

    realized = Some(cache.toMap)
  }

  def run() {
    require(realized.isDefined)

    val events = mutable.HashMap[MatrixExpr, CLEvent]()
    //    val outputEvents = mutable.HashMap[OutputCell, CLEvent]()

    def enqueue(e: MatrixExpr): Unit = {
      if (e.isInstanceOf[M.InputCell])
        return
      if (events.contains(e))
        return
      val re = realized.get(e)

      val targetBuf = re.out.buf

      def buf(e: MatrixExpr): BufferWithShape = {
        realized.get(e).out
      }

      def waitList(e: MatrixExpr*): Seq[CLEvent] = {
        e.filterNot(_.isInstanceOf[M.InputCell]).map(events)
      }

      e match {
        case M.Dotwise1(i, s, si) =>
          enqueue(i)
          val b = buf(i)
          // TODO!!!
          val kernel = program.createCLKernel("ew_cos").setArgs(b.buf, targetBuf)
          val eventsList = new CLEventList(waitList(i): _*)
          val thisEvent = new CLEventList(1)
          queue.put1DRangeKernel(kernel, 0, b.shape.sizeP, 64, eventsList, thisEvent)
          events += (e -> thisEvent.getEvent(0))
        case M.Prod(i1, i2) =>
          enqueue(i1)
          enqueue(i2)
          val b1 = buf(i1)
          val b2 = buf(i2)
          val b1shape = b1.shape.asInstanceOf[RealizedMatrix]
          val b2shape = b2.shape.asInstanceOf[RealizedMatrix]
          val kernel = program.createCLKernel("mmul").setArgs(b1.buf, b2.buf, Int.box(b1shape.colsP), targetBuf)

          val eventsList = new CLEventList(waitList(i1, i2): _*)
          val thisEvent = new CLEventList(1)
          queue.put2DRangeKernel(kernel, 0, 0, b1shape.rowsP, b2shape.colsP, 32, 16, eventsList, thisEvent)

          events += (e -> thisEvent.getEvent(0))
        case M.Transpose(i) =>
          enqueue(i)
          val b = buf(i)
          val shape = b.shape.asInstanceOf[RealizedMatrix]
          val kernel = program.createCLKernel("transpose").setArgs(targetBuf, b, Int.box(shape.rowsP), Int.box(shape.colsP))

          val eventsList = new CLEventList(waitList(i): _*)
          val thisEvent = new CLEventList(1)

          queue.put1DRangeKernel(kernel, 0, shape.sizeP, 64, eventsList, thisEvent)

          events += (e -> thisEvent.getEvent(0))
        case e@M.OutputCell(_, i) =>
          enqueue(i)
      }
    }

    //    for (in <- variables.values) {
    //      events += (in -> clContext.createUserEvent())
    //    }
    //
    for ((name, out) <- roots) {
      enqueue(out)
    }

    //    for (in <- variables.values) {
    //      events(in).asInstanceOf[CLUserEvent].setComplete()
    //    }
    //
    queue.finish()
  }
}

object ExprGraph {
  type FloatCLBuffer = CLBuffer[FloatBuffer]

  sealed trait RealizedShape {
    def size: Int
    def sizeP: Int
  }

  case class RealizedMatrix(rows: Int, cols: Int) extends RealizedShape {
    val size = rows * cols
    val rowsP = padTo(rows, 32)
    val colsP = padTo(cols, 32)
    val sizeP = rowsP * colsP
  }

  case class BufferWithShape(buf: FloatCLBuffer, shape: RealizedShape)

  private case class RealizedExpr(expr: MatrixExpr, out: BufferWithShape)

  private def padTo(s: Int, p: Int): Int = {
    ((s-1)/p + 1)*p
  }
}
