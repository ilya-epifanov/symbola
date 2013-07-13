package com.github.hajile.symbola.ad

import com.github.hajile.symbola.fn._
import scala.collection.immutable
import scala.collection.immutable.SortedMap

class SymbolicBackwardGradient {
  def matrix(gradientOf: MatrixExpr, wrt: Set[M.InputCell]): Map[M.InputCell, MatrixExpr] = {
    SortedMap()(M.InputCell.OrderingByName) ++
            SymbolicBackwardGradient.getAdjoint(gradientOf, wrt, M.Eye).mapValues {
              es =>
                M.Sum(es.toSeq)
            }
  }

  def scalar(gradientOf: ScalarExpr, wrt: Set[S.InputCell]): Map[S.InputCell, ScalarExpr] = {
    SortedMap()(S.InputCell.OrderingByName) ++
            SymbolicBackwardGradient.getAdjoint(gradientOf, wrt, S.One).mapValues {
              es =>
                S.Sum(es.toSeq: _*)
            }
  }
}

object SymbolicBackwardGradient {
  private def getAdjoint(node: MatrixExpr, wrt: Set[M.InputCell], seed: MatrixExpr): immutable.HashMap[M.InputCell, Set[MatrixExpr]] = {
    val in = MatrixExprHasInputs.inputs(node)
    val adjoints: Seq[immutable.HashMap[M.InputCell, Set[MatrixExpr]]] = in.collect {
      case i: M.InputCell if wrt.contains(i) =>
        immutable.HashMap[M.InputCell, Set[MatrixExpr]](i -> Set(node.grad(seed, i)))
      case i: M.InputCell =>
        immutable.HashMap[M.InputCell, Set[MatrixExpr]]()
      case e =>
        getAdjoint(e, wrt, node.grad(seed, e))
    }
    if (adjoints.isEmpty)
      immutable.HashMap[M.InputCell, Set[MatrixExpr]]()
    else
      adjoints.reduce {
        (a, b) =>
          a.merged(b) {
            case ((k1, v1), (_, v2)) =>
              k1 -> (v1 ++ v2)
          }
      }
  }

  private def getAdjoint(node: ScalarExpr, wrt: Set[S.InputCell], seed: ScalarExpr): immutable.HashMap[S.InputCell, Set[ScalarExpr]] = {
    val in = ScalarExprHasInputs.inputs(node)
    val adjoints: Seq[immutable.HashMap[S.InputCell, Set[ScalarExpr]]] = in.collect {
      case i: S.InputCell if wrt.contains(i) =>
        immutable.HashMap[S.InputCell, Set[ScalarExpr]](i -> Set(node.grad(seed, i)))
      case i: S.InputCell =>
        immutable.HashMap[S.InputCell, Set[ScalarExpr]]()
      case e =>
        getAdjoint(e, wrt, node.grad(seed, e))
    }
    if (adjoints.isEmpty)
      immutable.HashMap[S.InputCell, Set[ScalarExpr]]()
    else
      adjoints.reduce {
        (a, b) =>
          a.merged(b) {
            case ((k1, v1), (_, v2)) =>
              k1 -> (v1 ++ v2)
          }
      }
  }

}