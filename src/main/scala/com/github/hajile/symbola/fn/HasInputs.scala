package com.github.hajile.symbola.fn

trait HasInputs[E <: Expr[E]] {
  def inputs(expr: E): Seq[E]
  def map(expr: E, fn: (E) => E): E
}

object MatrixExprHasInputs extends HasInputs[MatrixExpr] {
  def inputs(expr: MatrixExpr) = expr match {
    case M.Sum(es) => es
    case M.Prod(e1, e2) => Seq(e1, e2)
    case M.Transpose(e) => Seq(e)
    case M.Dotwise1(e, _, _) => Seq(e)
    case i: M.InputCell => Seq()
    case M.Eye => Seq()
  }

  def map(expr: MatrixExpr, fn: (MatrixExpr) => MatrixExpr): MatrixExpr = expr match {
    case M.Sum(es) => M.Sum(es.map(fn))
    case M.Prod(e1, e2) => M.Prod(fn(e1), fn(e2))
    case M.Transpose(e) => M.Transpose(fn(e))
    case M.Dotwise1(e, s, si) => M.Dotwise1(fn(e), s, si)
    case i: M.InputCell => i
    case M.Eye => M.Eye
  }
}

object ScalarExprHasInputs extends HasInputs[ScalarExpr] {
  def inputs(expr: ScalarExpr) = expr match {
    case S.Sin(e) => Seq(e)
    case S.Cos(e) => Seq(e)
    case S.Sum(es: Seq[ScalarExpr]) => es
    case S.Mul(e1, e2) => Seq(e1, e2)
    case S.Div(e1, e2) => Seq(e1, e2)
    case S.Pow(e1, e2) => Seq(e1, e2)
    case S.Ln(e) => Seq(e)
    case S.Neg(e) => Seq(e)
    case S.Value(_) => Seq()
    case S.One => Seq()
    case S.Zero => Seq()
    case e: S.InputCell => Seq()
  }

  def map(expr: ScalarExpr, fn: (ScalarExpr) => ScalarExpr): ScalarExpr = expr match {
    case S.Sin(e) => S.Sin(fn(e))
    case S.Cos(e) => S.Cos(fn(e))
    case S.Sum(es: Seq[ScalarExpr]) => S.Sum(es.map(fn): _*)
    case S.Mul(e1, e2) => S.Mul(fn(e1), fn(e2))
    case S.Div(e1, e2) => S.Div(fn(e1), fn(e2))
    case S.Pow(e1, e2) => S.Pow(fn(e1), fn(e2))
    case S.Ln(e) => S.Ln(fn(e))
    case S.Neg(e) => S.Neg(fn(e))
    case e: S.Value => e
    case S.One => S.One
    case S.Zero => S.Zero
    case e: S.InputCell => e
  }
}

