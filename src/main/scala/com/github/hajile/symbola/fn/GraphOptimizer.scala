package com.github.hajile.symbola.fn

import scala.collection.mutable

object GraphOptimizer {
  def optimize[A](expr: Map[A, Expr]): Map[A, Expr] = {
    implicit val cache = mutable.HashMap[Expr, Expr]()
    expr.mapValues(forwardPass)
  }

  def forwardPass(e: Expr)(implicit cache: mutable.HashMap[Expr, Expr]): Expr = {
    cached(cached(e) match {
      case p@Prod(e1, e2) =>
        (forwardPass(e1), forwardPass(e2)) match {
          case (Eye, e2) =>
            e2
          case (e1, Eye) =>
            e1
          case (Transpose(e1), Transpose(e2)) =>
            forwardPass(Transpose(Prod(e2, e1)))
          case (e1, e2) => Prod(e1, e2)
        }
      case Sum(es) if es.size == 1 =>
        forwardPass(es(0))
      case e =>
        ExprHasInputs.map(e, forwardPass)
    })
  }

  private def cached(e: Expr)(implicit cache: mutable.HashMap[Expr, Expr]): Expr = {
    cache.get(e) match {
      case Some(existing) =>
        existing
      case None =>
        cache += (e -> e)
        e
    }
  }
}
