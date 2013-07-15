package com.github.hajile.symbola.fn

import com.github.hajile.symbola.ad.SymbolicBackwardGradient

sealed trait ScalarExpr extends Expr[ScalarExpr] {
  def apply() = eval()
  def eval(): Float
  def grad(seed: ScalarExpr, wrt: ScalarExpr): ScalarExpr
}

object ScalarExpr {
  case class InputCell(name: String) extends ScalarExpr {
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = sys.error("Cannot get here")
    def eval() = sys.error("Cannot get here")
    override def toString = name
  }

  object InputCell {
    implicit object OrderingByName extends Ordering[InputCell] {
      def compare(x: InputCell, y: InputCell) = x.name.compareTo(y.name)
    }
  }

  case class OutputCell(name: String, expr: ScalarExpr) extends ScalarExpr {
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = sys.error("Cannot get here")
    def eval() = expr()
  }

  object OutputCell {
    implicit object OrderingByName extends Ordering[OutputCell] {
      def compare(x: OutputCell, y: OutputCell) = x.name.compareTo(y.name)
    }
  }

  case object One extends ScalarExpr {
    def eval() = 1.0f
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = sys.error("Cannot get here")
    override def toString = "1"
  }

  case object Zero extends ScalarExpr {
    def eval() = 0.0f
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = sys.error("Cannot get here")
    override def toString = "0"
  }

  case class Value(v: Float) extends ScalarExpr {
    def eval() = v
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = sys.error("Cannot get here")
    override def toString = f"$v"
  }

  case class Cos(e: ScalarExpr) extends ScalarExpr {
    def eval() = math.cos(e()).toFloat
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = Neg(Mul(seed, Sin(e)))
    override def toString = f"cos($e)"
  }

  case class Sin(e: ScalarExpr) extends ScalarExpr {
    def eval() = math.sin(e()).toFloat
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = Mul(seed, Cos(e))
    override def toString = f"sin($e)"
  }

  case class Sum(e: ScalarExpr*) extends ScalarExpr {
    def eval() = e.map(_()).reduce((a, b) => a + b)
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = seed
    override def toString = s"(${e.mkString("+")})"
  }

  case class Mul(e1: ScalarExpr, e2: ScalarExpr) extends ScalarExpr {
    def eval() = e1() * e2()
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = wrt match {
      case `e1` =>
        Mul(seed, e2)
      case `e2` =>
        Mul(e1, seed)
    }
    override def toString = f"$e1*$e2"
  }

  case class Div(e1: ScalarExpr, e2: ScalarExpr) extends ScalarExpr {
    def eval() = e1() / e2()
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = wrt match {
      case `e1` =>
        Mul(seed, Div(One, e2))
      case `e2` =>
        Neg(Mul(seed, Div(e1, Mul(e2, e2))))
    }
    override def toString = f"$e1/$e2"
  }

  case class Neg(e: ScalarExpr) extends ScalarExpr {
    def eval() = -e()
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = Neg(seed)
    override def toString = f"-$e"
  }

  case class Pow(e1: ScalarExpr, e2: ScalarExpr) extends ScalarExpr {
    def eval() = math.pow(e1(), e2()).toFloat
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = wrt match {
      case `e1` =>
        Mul(seed, Mul(e2, Pow(e1, Sum(e2, Neg(One)))))
      case `e2` =>
        Mul(seed, Mul(Ln(e1), Pow(e1, e2)))
    }
    override def toString = f"$e1^$e2"
  }

  case class Ln(e: ScalarExpr) extends ScalarExpr {
    def eval() = math.log(e()).toFloat
    def grad(seed: ScalarExpr, wrt: ScalarExpr) = Mul(seed, Div(One, e))
    override def toString = f"ln($e)"
  }

  case class Grad(e: ScalarExpr, wrt: Set[S.InputCell]) {
    private val grads = new SymbolicBackwardGradient().scalar(e, wrt)
    def apply(wrt: S.InputCell): ScalarExpr = grads(wrt)
    override def toString = f"""∇($e|${wrt.mkString(",")})"""
  }
}