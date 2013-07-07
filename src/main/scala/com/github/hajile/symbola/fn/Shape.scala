package com.github.hajile.symbola.fn

sealed trait Shape {}

case class Matrix(rows: Int, cols: Int) extends Shape

case object Scalar extends Shape
