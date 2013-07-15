package com.github.hajile.symbola.fn

import com.github.hajile.symbola.ad.SymbolicBackwardGradient


sealed trait MatrixExpr extends Expr[MatrixExpr] {
  //  def eval(): SimpleMatrix
  //  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext): Unit
  def grad(seed: MatrixExpr, wrt: MatrixExpr): MatrixExpr
}

object MatrixExpr {

  case class InputCell(name: String /*, private var v: SimpleMatrix*/) extends MatrixExpr {
    def grad(seed: MatrixExpr, wrt: MatrixExpr) = sys.error("Cannot get here")
    override def toString = name
  }

  object InputCell {
    implicit object OrderingByName extends Ordering[InputCell] {
      def compare(x: InputCell, y: InputCell) = x.name.compareTo(y.name)
    }
  }

  case class OutputCell(name: String, expr: MatrixExpr) extends MatrixExpr {
    def grad(seed: MatrixExpr, wrt: MatrixExpr) = sys.error("Cannot get here")
  }

  object OutputCell {
    implicit object OrderingByName extends Ordering[OutputCell] {
      def compare(x: OutputCell, y: OutputCell) = x.name.compareTo(y.name)
    }
  }

  case class Value(value: SimpleMatrix) extends MatrixExpr {
    //  def apply(): SimpleMatrix = value
    def grad(seed: MatrixExpr, wrt: MatrixExpr) = sys.error("Cannot get here")
    override def toString = f"$value"
  }

  case object Eye extends MatrixExpr {
    def grad(seed: MatrixExpr, in: MatrixExpr) = sys.error("Cannot get here")
    override def toString = "I"
  }

  case object One extends MatrixExpr {
    def grad(seed: MatrixExpr, in: MatrixExpr) = sys.error("Cannot get here")
    override def toString = "[1]"
  }

  case object Zero extends MatrixExpr {
    def grad(seed: MatrixExpr, in: MatrixExpr) = sys.error("Cannot get here")
    override def toString = "[0]"
  }

  case class Sum(e: Seq[MatrixExpr]) extends MatrixExpr {
    //  def apply() = e.map(_()).reduce((a, b) => a + b)
    def grad(seed: MatrixExpr, wrt: MatrixExpr) = seed
    override def toString = s"(${e.mkString("+")})"
  }

  case class Prod(e1: MatrixExpr, e2: MatrixExpr) extends MatrixExpr {
    //  def apply() = {
    //    println("e1*e2")
    //    println(s"e1: ${e1()} ($e1)")
    //    println(s"e2: ${e2()} ($e2)")
    //    val r = e1() * e2()
    //    println(s"e1*e2 = $r")
    //    r
    //  }

    def grad(seed: MatrixExpr, wrt: MatrixExpr) = wrt match {
      case `e1` =>
        Prod(seed, Transpose(e2))
      case `e2` =>
        Prod(Transpose(e1), seed)
    }

    override def toString = f"$e1∙$e2"
  }

  case class Transpose(e: MatrixExpr) extends MatrixExpr {
    //  override def apply() = e().t
    def grad(seed: MatrixExpr, wrt: MatrixExpr) = Transpose(seed)
    override def toString = f"($e)ᵀ"
  }

  case class Dotwise1(e: MatrixExpr, s: ScalarExpr, in: S.InputCell) extends MatrixExpr {
    def grad(seed: MatrixExpr, wrt: MatrixExpr) = {
      val grad = S.Grad(s, Set(in))
      Dotwise1(e, grad(in), in)
    }
    override def toString = f"∘($e, $s)"
  }

  case class Grad(e: MatrixExpr, wrt: Set[M.InputCell]) {
    private val grads = new SymbolicBackwardGradient().matrix(e, wrt)
    def apply(wrt: M.InputCell): MatrixExpr = grads(wrt)
    override def toString = f"""∇($e|${wrt.mkString(",")})"""
  }
}