package com.github.hajile.symbola.cl

import breeze.linalg.DenseMatrix
import com.github.hajile.symbola.fn.ExprGraph.RealizedMatrix
import com.google.common.base.Charsets
import com.google.common.io.Resources
import com.jogamp.opencl.CLBuffer
import com.jogamp.opencl.CLMemory.Mem
import java.nio.FloatBuffer
import scala.util.Random

class OpenCLExample

object OpenCLExample extends OpenCLApp {
  //  val kernelName = if (args.length >= 2) args(1) else "mmultopt"

  println(s"Available local memory: ${device.getLocalMemSize}")

  val debug = true
  val tile = if (debug) 4 else 16
  val tileSize = tile * tile
  val sideN = if (debug) tile*3 else roundUpTo(1024, tile)
  val sideM = if (debug) tile*2 else roundUpTo(1024, tile)
  val sideP = if (debug) tile*4 else roundUpTo(1024, tile)

  val buf1 = ctx.createFloatBuffer(sideN / tile * sideP / tile * tileSize, Mem.ALLOCATE_BUFFER, Mem.READ_ONLY)
  val buf2 = ctx.createFloatBuffer(sideP / tile * sideM / tile * tileSize, Mem.ALLOCATE_BUFFER, Mem.READ_ONLY)
  val buf3 = ctx.createFloatBuffer(sideN / tile * sideM / tile * tileSize, Mem.ALLOCATE_BUFFER, Mem.READ_WRITE)

  val src = Resources.toString(classOf[OpenCLExample].getResource("kernels/mmul.cl"), Charsets.UTF_8)
  val program = ctx.createProgram(src)

  program.build("-cl-mad-enable")

  val kernel = program.createCLKernel("mmul")
  kernel.setArg(0, buf3)
  kernel.setArg(1, buf1)
  kernel.setArg(2, buf2)
  kernel.setArg(3, sideN)
  kernel.setArg(4, sideM)
  kernel.setArg(5, sideP)

  val rng = new Random(0)
  val ma = new DenseMatrix[Float](sideN, sideP)
  for (i <- 0 until sideN; j <- 0 until sideP) {
    ma.update(i, j, /*rng.nextFloat()*/ (i + 0.1*j).toFloat)
  }

  val mb = new DenseMatrix[Float](sideP, sideM)
  for (i <- 0 until sideP; j <- 0 until sideM) {
    mb.update(i, j, if (i != j) 0.5f else 0.1f/*rng.nextFloat()*/)
  }

  for (i <- 0 until (if (debug) 1 else 5)) {
    val mx1 = new MatrixFloatBuffer(buf1.getBuffer, RealizedMatrix(sideN, sideP))
    val mx2 = new MatrixFloatBuffer(buf2.getBuffer, RealizedMatrix(sideM, sideP))
//    val mx3 = new MatrixFloatBuffer(buf3.getBuffer, RealizedMatrix(sideN, sideM))
    for (i <- 0 until sideN; j <- 0 until sideP)
      mx1.put(i, j, ma(i, j))
    for (i <- 0 until sideP; j <- 0 until sideM)
      mx2.put(j, i, mb(i, j))

    //    for (bi <- 0 until sideN/tile; bj <- 0 until sideP/tile) {
    //      val b = roundUpTo(tile * tile * (bi * sideP / tile + bj), tileSize)
    //      for (i <- 0 until tile; j <- 0 until tile) {
    //        ptr1.put(b + i + j * tile, ma(bi * tile + i, bj * tile + j))
    //      }
    //    }
    //    for (bi <- 0 until sideP/tile; bj <- 0 until sideM/tile) {
    ////      val b = roundUpTo(tile * tile * (bi * sideM / tile + bj), tileSize)
    ////      for (i <- 0 until tile; j <- 0 until tile) {
    ////        ptr2.put(b + i + j * tile, mb(bi * tile + i, bj * tile + j))
    ////      }
    //      val b = roundUpTo((tile * tile) * (bi + bj * sideP / tile), tileSize)
    //      for (i <- 0 until tile; j <- 0 until tile) {
    //        ptr2.put(b + i * tile + j, mb(bi * tile + i, bj * tile + j))
    //      }
    //    }
    if (debug) {
      println(s"----- A -----\n\n$ma\n\n")
      println(s"-- A bytes --\n\n${dumpBuffer(buf1)}\n\n")
      println(s"----- B -----\n\n$mb\n\n")
      println(s"-- B bytes --\n\n${dumpBuffer(buf2)}\n\n")
    }

    q.putWriteBuffer(buf1, true)
    q.putWriteBuffer(buf2, true)

    q.finish()
    val began = System.nanoTime()
    q.put1DRangeKernel(kernel, 0, sideN * sideM / 4, tile * tile / 4)
    q.putReadBuffer(buf3, true)
    q.finish()

    val duration = System.nanoTime() - began
    println(f"Kernel executed in ${duration / 1000000.0}%.2fms")

    if (debug) {
      val cpuBegan = System.nanoTime()
      val mcref = ma * mb
      val cpuDuration = System.nanoTime() - cpuBegan
      println(f"CPU naïve algorithm executed in ${cpuDuration / 1000000.0}%.2fms")

      val ret = buf3.getBuffer
      val mccl = new MatrixFloatBuffer(ret, RealizedMatrix(sideN, sideM)).toDenseMatrix
      //      val mccl = new DenseMatrix[Float](sideN, sideM)
      //      for (i <- 0 until sideN; j <- 0 until sideM)
      //        mccl.update(i, j, ret.get(i + j * sideN))

      //      for (bi <- 0 until sideN/tile; bj <- 0 until sideM/tile) {
      //        val b = roundUpTo((tile * tile) * (bi * sideM / tile + bj), tileSize)
      //        for (i <- 0 until tile; j <- 0 until tile) {
      //          mccl.update(bi * tile + i, bj * tile + j, ret.get(b + i + j * tile))
      //        }
      //      }

      println(s"----- C (CL) -----\n\n${mccl.toString(16, 200)}\n\n")
      println(s"----- C bytes ----\n\n${dumpBuffer(buf3)}\n\n")

      println(s"----- C (ref) -----\n\n$mcref\n\n")

      //    for (j <- 0 until ret.limit()) {
      //      val x = j % side
      //      val y = j / side
      //      val r = ret.get(x + y * side)
      //      println(s"[$x][$y] : $r")
      //    }
    }
  }

  def roundUpTo(v: Int, modulo: Int): Int = {
    (ifloordiv(v - 1, modulo) + 1) * modulo
  }

  def ifloordiv(n: Int, d: Int): Int = {
    if (n >= 0) n / d
    else ~(~n / d)
  }


  def dumpBuffer(ptr: CLBuffer[FloatBuffer]): String = {
    val buf = ptr.getBuffer
    val str = StringBuilder.newBuilder
    for (i <- 0 until buf.limit()) {
      if (i != 0)
        str.append(' ')
      str.append(buf.get(i))
    }
    str.toString()
  }
}
