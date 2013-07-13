package com.github.hajile.symbola.cl.kernels

case class DotwiseKernel(name: String, out: String, input: Seq[String], ops: Seq[Op])

sealed trait Op {
  def out: String
}

case class ScalarOp(name: String, out: String, args: String*) extends Op

case class ConstOp(out: String, value: String) extends Op
