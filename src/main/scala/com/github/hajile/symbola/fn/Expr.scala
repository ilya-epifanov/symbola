package com.github.hajile.symbola.fn

import com.github.hajile.symbola.ad.BackwardGradient.BackwardContext
import breeze.numerics
import breeze.linalg.DenseMatrix

sealed trait Expr {
  def apply(): SimpleMatrix
  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext): Unit
  def backwardsExpr(seed: Expr, in: Expr): Expr
}

//case class InputCell(name: String, shape: Shape) extends Expr
//
//case class OutputCell(name: String, shape: Shape) extends Expr

case class Input(name: String, private var v: SimpleMatrix) extends Expr {
  def get: SimpleMatrix = v

  def set(value: SimpleMatrix): Unit = v = value

  def apply() = v

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) = ctx.contribute(this, c)

  override def toString = name

  def backwardsExpr(seed: Expr, in: Expr) = sys.error("Cannot get here")
}

object Input {
  implicit object InputOrderingByName extends Ordering[Input] {
    def compare(x: Input, y: Input) = x.name.compareTo(y.name)
  }
}

case class Value(value: SimpleMatrix) extends Expr {
  def apply(): SimpleMatrix = value

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) = sys.error("Cannot get here")

  override def toString = f"$value"

  def backwardsExpr(seed: Expr, in: Expr) = sys.error("Cannot get here")
}

case class Eye(size: Int) extends Expr {
  private val value = DenseMatrix.eye[Double](size)

  def apply(): SimpleMatrix = value

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) = sys.error("Cannot get here")

  override def toString = "I" + size.toString.map(c => (c - '0' + '₀').toChar).mkString

  def backwardsExpr(seed: Expr, in: Expr) = sys.error("Cannot get here")
}

case class Sin(e: Expr) extends Expr {
  def apply(): SimpleMatrix = numerics.sin(e())

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) {
    e.backwards(c :* numerics.cos(e()))
  }

  override def toString = f"sin($e)"

  def backwardsExpr(seed: Expr, in: Expr) = {
    ElemwiseMul(seed, Cos(e))
  }
}

case class Id(e: Expr) extends Expr {
  def apply() = e()

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) {
    e.backwards(c)
  }

  override def toString = f"($e)"

  def backwardsExpr(seed: Expr, in: Expr) = {
    seed
  }
}

case class Cos(e: Expr) extends Expr {
  def apply() = numerics.cos(e())

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) {
    e.backwards(c :* numerics.sin(e()) * -1.0)
  }

  override def toString = f"cos($e)"

  def backwardsExpr(seed: Expr, in: Expr) = {
    Neg(ElemwiseMul(seed, Sin(e)))
  }
}

case class Sum(e: Seq[Expr]) extends Expr {
  def apply() = e.map(_()).reduce((a, b) => a + b)

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) {
    e.foreach(_.backwards(c))
  }

  def backwardsExpr(seed: Expr, in: Expr) = {
    seed
  }

  override def toString = s"(${e.mkString("+")})"
}

case class ElemwiseMul(e1: Expr, e2: Expr) extends Expr {
  override def apply() = e1() :* e2()

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) {
    e1.backwards(c :* e2())
    e2.backwards(c :* e1())
  }

  def backwardsExpr(seed: Expr, in: Expr) = {
    in match {
      case `e1` => ElemwiseMul(seed, e2)
      case `e2` => ElemwiseMul(seed, e1)
    }
  }

  override def toString = s"$e1∘$e2"
}

case class Prod(e1: Expr, e2: Expr) extends Expr {
  def apply() = {
//    println("e1*e2")
//    println(s"e1: ${e1()} ($e1)")
//    println(s"e2: ${e2()} ($e2)")
    val r = e1() * e2()
//    println(s"e1*e2 = $r")
    r
  }

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) {
    e2.backwards(e1().t * c)
    e1.backwards(c * e2().t)
  }

  def backwardsExpr(seed: Expr, in: Expr) = {
    in match {
      case `e1` =>
        Prod(seed, Transpose(e2))
      case `e2` =>
        Prod(Transpose(e1), seed)
    }
  }

  override def toString = f"$e1∙$e2"
}

case class Transpose(e: Expr) extends Expr {
  override def apply() = e().t

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) {
    e.backwards(c.t)
  }

  def backwardsExpr(seed: Expr, in: Expr) = {
    Transpose(seed)
  }

  override def toString = f"($e)ᵀ"
}

case class Neg(e: Expr) extends Expr {
  def apply() = e().map(-_)

  def backwards(c: SimpleMatrix)(implicit ctx: BackwardContext) {
    e.backwards(c * -1.0)
  }

  def backwardsExpr(seed: Expr, in: Expr) = {
    Neg(seed)
  }

  override def toString = f"-$e"
}
