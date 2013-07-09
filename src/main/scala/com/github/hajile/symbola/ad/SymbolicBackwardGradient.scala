package com.github.hajile.symbola.ad

import com.github.hajile.symbola.fn.ExprHasInputs._
import com.github.hajile.symbola.fn.{InputCell, Eye, Expr, Sum}
import scala.collection.immutable
import scala.collection.immutable.SortedMap

class SymbolicBackwardGradient {
  def adjointTreesFor(gradientOf: Expr, wrt: Set[InputCell]): Map[InputCell, Expr] = {
    SortedMap()(InputCell.OrderingByName) ++
            SymbolicBackwardGradient.getAdjoint(gradientOf, wrt, Eye).mapValues {
              es =>
                Sum(es.toSeq)
            }
  }
}

object SymbolicBackwardGradient {
  private def getAdjoint(node: Expr, wrt: Set[InputCell], seed: Expr): immutable.HashMap[InputCell, Set[Expr]] = {
    val in = inputs(node)
    val adjoints: Seq[immutable.HashMap[InputCell, Set[Expr]]] = in.collect {
      case i: InputCell if wrt.contains(i) =>
        immutable.HashMap[InputCell, Set[Expr]](i -> Set(node.backwardsExpr(seed, i)))
      case i: InputCell =>
        immutable.HashMap[InputCell, Set[Expr]]()
      case e =>
        getAdjoint(e, wrt, node.backwardsExpr(seed, e))
    }
    if (adjoints.isEmpty)
      immutable.HashMap[InputCell, Set[Expr]]()
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