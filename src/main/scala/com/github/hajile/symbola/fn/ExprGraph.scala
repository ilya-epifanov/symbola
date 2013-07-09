package com.github.hajile.symbola.fn

import com.nativelibs4java.opencl.{CLEvent, CLMem, CLContext, CLBuffer}
import scala.collection.mutable
import com.nativelibs4java.opencl.CLMem.Usage
import com.nativelibs4java.util.IOUtils
import com.github.hajile.symbola.cl.OpenCLExample
import org.bridj.Pointer

class ExprGraph(val clContext: CLContext) {

  import ExprGraph._

  private val queue = clContext.createDefaultQueue()
  private val program = clContext.createProgram(IOUtils.readText(classOf[OpenCLExample].getResource("kernels/kernels.cl")))
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

  def put(name: String, shape: RealizedShape, buf: Pointer[java.lang.Float]) {
    require(buf.getValidElements == shape.size)

    val clBuf = alloc(shape.size)(variables(name))
    inputValues += (name -> BufferWithShape(clBuf, shape))

    clBuf.write(queue, buf, false).waitFor()
  }

  def get(name: String, buf: Pointer[java.lang.Float]) {
    val out = realized.get(roots(name).expr).out
    require(buf.getValidElements == out.shape.size)

    out.buf.read(queue, buf, false).waitFor()
  }

  def alloc(size: Int, kind: CLMem.Usage = Usage.InputOutput)(implicit e: Expr): FloatCLBuffer = {
    println(f"Allocating CL Buffer with size ${size * 4 / 1048576.0}%.2fMiB for expression $e")
    clContext.createFloatBuffer(kind, size)
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
          val kernel = program.createKernel("ew_cos", b.buf, targetBuf)
          events += (e -> kernel.enqueueNDRange(queue, Array(b.shape.size), Array(64), waitList(i): _*))
        case Prod(i1, i2) =>
          enqueue(i1)
          enqueue(i2)
          val b1 = buf(i1)
          val b2 = buf(i2)
          val b1shape = b1.shape.asInstanceOf[RealizedMatrix]
          val b2shape = b2.shape.asInstanceOf[RealizedMatrix]
          val kernel = program.createKernel("mmult", b1.buf, b2.buf, Int.box(b1shape.cols), targetBuf)
          events += (e -> kernel.enqueueNDRange(queue, Array(b1shape.rows, b2shape.cols), Array(32, 16), waitList(i1, i2): _*))
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
  type FloatCLBuffer = CLBuffer[java.lang.Float]

  sealed trait RealizedShape {
    def size: Int
  }

  case class RealizedMatrix(rows: Int, cols: Int) extends RealizedShape {
    def size = rows * cols
  }

  case class BufferWithShape(buf: FloatCLBuffer, shape: RealizedShape)

  private case class RealizedExpr(expr: Expr, out: BufferWithShape)

}
