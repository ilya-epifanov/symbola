package com.github.hajile.symbola.fn

trait Expr[E <: Expr[E]] {
  def grad(seed: E, wrt: E): E
}
