package com.github.hajile.symbola.fn

import scala.collection.mutable
import com.github.hajile.symbola.cl.OpenCLExample
import com.jogamp.opencl.{CLEventList, CLEvent, CLBuffer, CLContext}
import com.google.common.io.Resources
import com.google.common.base.Charsets
import java.nio.FloatBuffer
import com.jogamp.opencl.CLMemory.Mem

class ExprGraph(val clContext: CLContext) {

  import ExprGraph._

  private val queue = clContext.getMaxFlopsDevice.createCommandQueue()
  private val program = clContext.createProgram(Resources.toString(classOf[OpenCLExample].getResource("kernels/kernels.cl"), Charsets.UTF_8))
  //  program.setFastRelaxedMath()
  //  program.setUnsafeMathOptimizations()
  program.build()

  private val variables = mutable.HashMap[String, InputCell]()
  private val roots = mutable.HashMap[String, OutputCell]()

  private val inputValues = mutable.HashMap[String, BufferWithShape]()
  //  private val outputValues = mutable.HashMap[String, BufferWithShape]()

  private var realized: Option[Map[Expr, RealizedExpr]] = None

  def in(name: String, shape: Shape): InputCell = {
    require(!variables.contains(name))

    val c = InputCell(name, shape)
    variables += (name -> c)
    c
  }

  def out(name: String, expr: Expr): OutputCell = {
    require(!roots.contains(name))

    val c = OutputCell(name, expr)
    roots += (name -> c)
    c
  }

  def inValue(name: String): BufferWithShape = inputValues(name)

  def getIn(name: String, shape: RealizedShape): FloatBuffer = {
    val clBuf = alloc(shape.size)(variables(name))
    inputValues += (name -> BufferWithShape(clBuf, shape))
    clBuf.getBuffer
  }

  def writeIn(name: String) {
    queue.putWriteBuffer(inputValues(name).buf, true)
  }

  def alloc(size: Int)(implicit e: Expr): FloatCLBuffer = {
    println(f"Allocating CL Buffer with size ${size * 4 / 1048576.0}%.2fMiB for expression $e")
    clContext.createFloatBuffer(size, Mem.READ_WRITE)
  }

  def realize() {
    require(variables.keySet == inputValues.keySet, "Some inputs aren't initialized")

    val cache = mutable.HashMap[Expr, RealizedExpr]()

    def realizeExpr(e: Expr): RealizedExpr = {
      def r(implicit e: Expr, shape: RealizedShape): RealizedExpr = {
        require(!cache.contains(e))
        val ret = RealizedExpr(e, BufferWithShape(alloc(shape.size), shape))
        cache += e -> ret
        ret
      }

      if (cache.contains(e))
        cache(e)
      else e match {
        case c@InputCell(name, shape) =>
          require(variables.contains(name), "Where did you get that variable? It doesn't belong to this ExprGraph")
          val ret = RealizedExpr(c, inputValues(name))
          cache += e -> ret
          ret
        //          r(c, inputValues(name).shape)
        case e@Cos(i) =>
          r(e, realizeExpr(i).out.shape)
        case e@ElemwiseMul(i1, i2) =>
          val shape1 = realizeExpr(i1).out.shape
          val shape2 = realizeExpr(i2).out.shape
          require(shape1 == shape2)
          r(e, shape1)
        case e@Neg(i) =>
          r(e, realizeExpr(i).out.shape)
        case e@OutputCell(_, i) =>
          val ret = RealizedExpr(e, realizeExpr(i).out)
          cache += e -> ret
          ret
        case e@Prod(i1, i2) =>
          val shape1 = realizeExpr(i1).out.shape.asInstanceOf[RealizedMatrix]
          val shape2 = realizeExpr(i2).out.shape.asInstanceOf[RealizedMatrix]
          require(shape1.cols == shape2.rows)
          r(e, RealizedMatrix(shape1.rows, shape2.cols))
        case e@Sin(i) =>
          r(e, realizeExpr(i).out.shape)
        case e@Sum(is) =>
          val shapes = is.map(realizeExpr(_).out.shape)
          require(shapes.forall(_ == shapes(0)))
          r(e, shapes(0))
        case e@Transpose(i) =>
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

    val events = mutable.HashMap[Expr, CLEvent]()
    //    val outputEvents = mutable.HashMap[OutputCell, CLEvent]()

    def enqueue(e: Expr) {
      //      if (e.isInstanceOf[InputCell])
      //        return
      if (events.contains(e))
        return
      val re = realized.get(e)

      val targetBuf = re.out.buf

      def buf(e: Expr): BufferWithShape = {
        realized.get(e).out
      }

      def waitList(e: Expr*): Seq[CLEvent] = {
        e.filterNot(_.isInstanceOf[InputCell]).map(events)
      }

      e match {
        case Cos(i) =>
          enqueue(i)
          val b = buf(i)
          val kernel = program.createCLKernel("ew_cos").setArgs(b.buf, targetBuf)
          val eventsList = new CLEventList(events(i))
          val thisEvent = new CLEventList(1)
          queue.put1DRangeKernel(kernel, 0, b.shape.size, 64, eventsList, thisEvent)
          events += (e -> thisEvent.getEvent(0))
        case Prod(i1, i2) =>
          enqueue(i1)
          enqueue(i2)
          val b1 = buf(i1)
          val b2 = buf(i2)
          val b1shape = b1.shape.asInstanceOf[RealizedMatrix]
          val b2shape = b2.shape.asInstanceOf[RealizedMatrix]
          val kernel = program.createCLKernel("mmult").setArgs(b1.buf, b2.buf, Int.box(b1shape.cols), targetBuf)

          val eventsList = new CLEventList(waitList(i1, i2): _*)
          val thisEvent = new CLEventList(1)
          queue.put2DRangeKernel(kernel, 0, 0, b1shape.rows, b2shape.cols, 32, 16, eventsList, thisEvent)

          events += (e -> thisEvent.getEvent(0))
        case e@OutputCell(_, i) =>
          enqueue(i)
        case e: InputCell =>
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
  }

  case class RealizedMatrix(rows: Int, cols: Int) extends RealizedShape {
    def size = rows * cols
  }

  case class BufferWithShape(buf: FloatCLBuffer, shape: RealizedShape)

  private case class RealizedExpr(expr: Expr, out: BufferWithShape)

}
