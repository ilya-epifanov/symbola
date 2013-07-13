package com.github.hajile.symbola.fn

import java.io.FileWriter
import scala.collection.immutable.HashSet

object DotVisualizer {
  def visualize(name: String, e: MatrixExpr*) {
    val g = visualize(e: _*)
    val filename = name + ".dot"
    val out = new FileWriter(filename)
    out.write(g)
    out.close()

    new ProcessBuilder("/usr/local/bin/dot", filename, "-Tsvg", "-o", name + ".svg").start().waitFor()
  }

  def visualize(e: MatrixExpr*): String = {
    "digraph G {\n\t" +
            "graph [size=7.25]\n\t" +
            e.map { n =>
              nodeName(n) + "[color = red, shape = rect]"
            }.mkString(";\n\t") +
            e.flatMap(collectEdges).map({
              case (n, i) =>
                nodeName(n) + " -> " + nodeName(i)
            }).toSet.mkString(";\n\t") +
            ";\n}"
  }

  private def nodeName(e: MatrixExpr): String = {
    "\"" + e.toString + /*"@" + Integer.toHexString(System.identityHashCode(e)) +*/ "\""
  }

  private def collectEdges(e: MatrixExpr): Set[(MatrixExpr, MatrixExpr)] = {
    var ret = HashSet[(MatrixExpr, MatrixExpr)]()
    for (i <- MatrixExprHasInputs.inputs(e)) {
      ret += (e -> i)
      ret ++= collectEdges(i)
    }
    ret
  }
}
