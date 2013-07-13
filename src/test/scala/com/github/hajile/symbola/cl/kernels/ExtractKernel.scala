package com.github.hajile.symbola.cl.kernels

trait ExtractKernel {
  implicit def string2ExtractKernelStringOps(str: String) = new ExtractKernel.ExtractKernelStringOps(str)
}

object ExtractKernel {
  implicit class ExtractKernelStringOps(val program: String) extends AnyVal {
    def extractKernel(): String = {
      program.drop(program.indexOfSlice("__kernel ")).trim
    }
  }
}