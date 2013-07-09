package com.github.hajile.symbola.fn

import com.nativelibs4java.opencl.{CLUserEvent, CLEvent, CLMem, CLContext, CLBuffer}
import scala.collection.mutable
import com.nativelibs4java.opencl.CLMem.Usage
import com.nativelibs4java.util.IOUtils
import com.github.hajile.symbola.cl.OpenCLExample

class ExprGraph(val clContext: CLContext) {
  import ExprGraph._

  private val queue = clContext.createDefaultOutOfOrderQueueIfPossible()
  private val program = clContext.createProgram(IOUtils.readText(classOf[OpenCLExample].getResource("kernels/kernels.cl")))
  program.setFastRelaxedMath()
  program.setUnsafeMathOptimizations()
  program.build()

  private val variables = mutable.HashMap[String, InputCell]()
  private val roots = mutable.HashMap[String, OutputCell]()

  private val inputValues = mutable.HashMap[String, BufferWithShape]()
  private val outputValues = mutable.HashMap[String, BufferWithShape]()

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

  def outValue(name: String): BufferWithShape = outputValues(name)

  def alloc(size: Int, kind: CLMem.Usage = Usage.InputOutput): FloatCLBuffer = {
    clContext.createFloatBuffer(kind, size)
  }

  def realize() {
    require(variables.keySet == inputValues.keySet, "Some inputs aren't initialized")

    val cache = mutable.HashMap[Expr, RealizedExpr]()

    def realizeExpr(e: Expr): RealizedExpr = {
      def r(e: Expr, shape: RealizedShape): RealizedExpr = {
        val ret = RealizedExpr(e, BufferWithShape(alloc(shape.size), shape))
        cache += e -> ret
        ret
      }

      if (cache.contains(e))
        cache(e)
      else e match {
        case c@InputCell(name, shape) =>
          require(variables.contains(name), "Where did you get that variable? It doesn't belong to this ExprGraph")
          r(c, inputValues(name).shape)
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
          r(e, realizeExpr(i).out.shape)
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
    val outputEvents = mutable.HashMap[OutputCell, CLEvent]()

    def enqueue(e: Expr) {
      if (events.contains(e))
        return
      val re = realized.get(e)

      val targetBuf = re.out.buf

      def buf(e: Expr): BufferWithShape = {
        realized.get(e).out
      }

      e match {
        case Cos(i) =>
          enqueue(i)
          val b = buf(i)
          val kernel = program.createKernel("cos", b.buf, targetBuf)
          kernel.enqueueNDRange(queue, Array(b.shape.size), events(i))
        case e@OutputCell(_, i) =>
          enqueue(i)
      }
    }

    for ((name, in) <- variables) {
      events += (in -> clContext.createUserEvent())
    }

    for ((name, out) <- roots) {
      enqueue(out)
    }

    for (in <- variables.values) {
      events(in).asInstanceOf[CLUserEvent].setComplete()
    }


  }
}

object ExprGraph {
  type FloatCLBuffer = CLBuffer[java.lang.Float]

  private sealed trait RealizedShape {
    def size: Int
  }

  private case class RealizedMatrix(rows: Int, cols: Int) extends RealizedShape{
    def size = rows * cols
  }

  private case class BufferWithShape(buf: FloatCLBuffer, shape: RealizedShape)

  private case class RealizedExpr(expr: Expr, out: BufferWithShape)
}
