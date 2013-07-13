package com.github.hajile.symbola.fn

import scala.collection.mutable

object GraphOptimizer {
  def optimizeMatrix[A](expr: Map[A, MatrixExpr]): Map[A, MatrixExpr] = {
    implicit val cache = mutable.HashMap[MatrixExpr, MatrixExpr]()
    expr.mapValues(forwardPass)
  }

  def forwardPass(e: MatrixExpr)(implicit cache: mutable.HashMap[MatrixExpr, MatrixExpr]): MatrixExpr = {
    def cached(e: MatrixExpr)(implicit cache: mutable.HashMap[MatrixExpr, MatrixExpr]): MatrixExpr = {
      cache.get(e) match {
        case Some(existing) =>
          existing
        case None =>
          cache += (e -> e)
          e
      }
    }

    cached(cached(e) match {
      case p@M.Prod(e1, e2) =>
        (forwardPass(e1), forwardPass(e2)) match {
          case (M.Eye, e2) =>
            e2
          case (e1, M.Eye) =>
            e1
          case (M.Transpose(e1), M.Transpose(e2)) =>
            forwardPass(M.Transpose(M.Prod(e2, e1)))
          case (e1, e2) => M.Prod(e1, e2)
        }
      case M.Sum(es) if es.size == 1 =>
        forwardPass(es(0))
      case M.Dotwise1(e, s, si) =>
        M.Dotwise1(forwardPass(e), optimizeScalar(s), si)
      case e =>
        MatrixExprHasInputs.map(e, forwardPass)
    })
  }

  def optimizeScalar[A](expr: ScalarExpr): ScalarExpr = {
    implicit val cache = mutable.HashMap[ScalarExpr, ScalarExpr]()
    forwardPass(expr)
  }

  def optimizeScalar[A](expr: Map[A, ScalarExpr]): Map[A, ScalarExpr] = {
    implicit val cache = mutable.HashMap[ScalarExpr, ScalarExpr]()
    expr.mapValues(forwardPass)
  }

  def forwardPass(e: ScalarExpr)(implicit cache: mutable.HashMap[ScalarExpr, ScalarExpr]): ScalarExpr = {
    def cached(e: ScalarExpr)(implicit cache: mutable.HashMap[ScalarExpr, ScalarExpr]): ScalarExpr = {
      cache.get(e) match {
        case Some(existing) =>
          existing
        case None =>
          cache += (e -> e)
          e
      }
    }

    cached(cached(e) match {
      case s@S.Mul(e1, e2) =>
        (forwardPass(e1), forwardPass(e2)) match {
          case (S.One, e2) =>
            e2
          case (e1, S.One) =>
            e1
          case (S.Zero, _) =>
            S.Zero
          case (_, S.Zero) =>
            S.Zero
          case _ => s
        }
      case S.Sum(es: Seq[ScalarExpr]) if es.size == 1 =>
        forwardPass(es(0))
      case S.Sum(es: Seq[ScalarExpr]) =>
        forwardPass(S.Sum(es.map(forwardPass).filter(_ != S.Zero): _*))
      case e =>
        ScalarExprHasInputs.map(e, forwardPass)
    })
  }
}
