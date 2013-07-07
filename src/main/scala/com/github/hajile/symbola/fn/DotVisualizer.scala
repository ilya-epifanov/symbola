package com.github.hajile.symbola.fn

import java.io.FileWriter

object DotVisualizer {
  def visualize(name: String, e: Expr*) {
    val g = visualize(e: _*)
    val filename = name + ".dot"
    val out = new FileWriter(filename)
    out.write(g)
    out.close()

    new ProcessBuilder("/usr/local/bin/dot", filename, "-Tsvg", "-o", name + ".svg").start().waitFor()
  }

  def visualize(e: Expr*): String = {
    "digraph G {\n\t" +
            "graph [size=7.25]\n\t" +
            e.map { n =>
              nodeName(n) + "[color = red, shape = rect]"
            }.mkString(";\n\t") +
            e.flatMap(collectEdges).map({
              case (n, i) =>
                nodeName(n) + " -> " + nodeName(i)
            }).mkString(";\n\t") +
            ";\n}"
  }

  private def nodeName(e: Expr): String = {
    "\"" + e.toString + "\""
  }

  private def collectEdges(e: Expr): Set[(Expr, Expr)] = {
    var ret = Set[(Expr, Expr)]()
    for (i <- ExprHasInputs.inputs(e)) {
      ret += (e -> i)
      ret ++= collectEdges(i)
    }
    ret
  }
}
