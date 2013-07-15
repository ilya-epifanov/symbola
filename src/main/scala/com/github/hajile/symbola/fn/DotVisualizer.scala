package com.github.hajile.symbola.fn

import java.io.FileWriter
import scala.collection.immutable.HashSet

object DotVisualizer {
  def visualize[E <: Expr[E] : HasInputs](name: String, e: E*) {
    val g = visualize(e: _*)
    val filename = name + ".dot"
    val out = new FileWriter(filename)
    out.write(g)
    out.close()

    new ProcessBuilder("/usr/local/bin/dot", filename, "-Tsvg", "-o", name + ".svg").start().waitFor()
  }

  def visualize[E <: Expr[E] : HasInputs](e: E*): String = {
    "digraph G {\n\t" +
            "graph [size=7.25]\n\t" +
            e.map { n =>
              nodeName(n) + "[color = red, shape = rect]"
            }.mkString(";\n\t") +
            e.flatMap(collectEdges[E]).map({
              case (n, i) =>
                nodeName(n) + " -> " + nodeName(i)
            }).toSet.mkString(";\n\t") +
            ";\n}"
  }

  private def nodeName[E <: Expr[E] : HasInputs](e: E): String = {
    "\"" + e.toString + /*"@" + Integer.toHexString(System.identityHashCode(e)) +*/ "\""
  }

  private def collectEdges[E <: Expr[E] : HasInputs](e: E): Set[(E, E)] = {
    var ret = HashSet[(E, E)]()
    for (i <- implicitly[HasInputs[E]].inputs(e)) {
      ret += (e -> i)
      ret ++= collectEdges(i)
    }
    ret
  }
}
