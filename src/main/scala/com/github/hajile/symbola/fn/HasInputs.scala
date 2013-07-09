package com.github.hajile.symbola.fn

trait HasInputs[E <: Expr] {
  def inputs(expr: E): Seq[Expr]
  def map(expr: E, fn: (Expr) => Expr): Expr
}

object ExprHasInputs extends HasInputs[Expr] {
  def inputs(expr: Expr) = expr match {
    case Sin(e) => Seq(e)
    case Cos(e) => Seq(e)
    case Sum(es) => es
    case Prod(e1, e2) => Seq(e1, e2)
    case Neg(e) => Seq(e)
    case Transpose(e) => Seq(e)
    case ElemwiseMul(e1, e2) => Seq(e1, e2)
    case _ => Seq()
  }

  def map(expr: Expr, fn: (Expr) => Expr): Expr = expr match {
    case Sin(e) => Sin(fn(e))
    case Cos(e) => Cos(fn(e))
    case Sum(es) => Sum(es.map(fn))
    case Prod(e1, e2) => Prod(fn(e1), fn(e2))
    case Neg(e) => Neg(fn(e))
    case Transpose(e) => Transpose(fn(e))
    case ElemwiseMul(e1, e2) => ElemwiseMul(fn(e1), fn(e2))
//    case Id(e) => Id(fn(e))
    case e => e
  }
}
