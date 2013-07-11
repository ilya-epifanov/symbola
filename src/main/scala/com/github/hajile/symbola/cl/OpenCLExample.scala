package com.github.hajile.symbola.cl

import breeze.linalg.DenseMatrix
import com.google.common.base.Charsets
import com.google.common.io.Resources
import com.jogamp.opencl.CLDevice.Type
import com.jogamp.opencl.CLMemory.Mem
import com.jogamp.opencl.util.Filter
import com.jogamp.opencl.{CLBuffer, CLContext, CLPlatform}
import java.nio.FloatBuffer
import scala.util.Random

class OpenCLExample

object OpenCLExample extends App {
  val platformString = if (args.length >= 1) args(0).toLowerCase else ""
  val kernelName = if (args.length >= 2) args(1) else "mmultopt"

  for (p <- CLPlatform.listCLPlatforms()) {
    println("Platform: " + p.getName)
    for (d <- p.listCLDevices(Type.ALL)) {
      println(s"  Device: ${d.getName} [${d.isAvailable}]")
    }
  }

  val device = CLPlatform.getDefault(new Filter[CLPlatform] {
    def accept(item: CLPlatform) = {
      val name = item.getName
      name.toLowerCase.contains(platformString)
    }
  }).getMaxFlopsDevice

  println(s"Using device: $device")
  val ctx = CLContext.create(device)

  val q = device.createCommandQueue()

  val debug = false
  val sideN = if (debug) 3 else 1024
  val sideM = if (debug) 4 else 1024
  val sideP = if (debug) 2 else 1024
  val tile = 16

  val buf1 = ctx.createFloatBuffer(sideN * sideP, Mem.ALLOCATE_BUFFER)
  val buf2 = ctx.createFloatBuffer(sideP * sideM, Mem.ALLOCATE_BUFFER)
  val buf3 = ctx.createFloatBuffer(sideN * sideM, Mem.ALLOCATE_BUFFER)

  val src = Resources.toString(classOf[OpenCLExample].getResource("kernels/kernels.cl"), Charsets.UTF_8)
  val program = ctx.createProgram(src)

//  program.setFastRelaxedMath()
//  program.setMadEnable()
//  program.setUnsafeMathOptimizations()

  program.build("-cl-mad-enable")

  val kernel = program.createCLKernel("mmultopt")
  kernel.setArg(0, buf1)
  kernel.setArg(1, buf2)
  kernel.setArg(2, sideN)
  kernel.setArg(3, sideM)
  kernel.setArg(4, sideP)
  kernel.setArg(5, buf3)

  val rng = new Random(0)
  val ma = new DenseMatrix[Float](sideN, sideP)
  for (i <- 0 until sideN; j <- 0 until sideP)
    ma.update(i, j, rng.nextFloat())

  val mb = new DenseMatrix[Float](sideP, sideM)
  for (i <- 0 until sideP; j <- 0 until sideM)
    mb.update(i, j, rng.nextFloat())

  if (debug) {
    println(s"----- A -----\n\n$ma\n\n")
    println(s"----- B -----\n\n$mb\n\n")
  }

  for (i <- 0 until (if (debug) 1 else 5)) {
    val ptr1 = buf1.getBuffer
    val ptr2 = buf2.getBuffer
    for (i <- 0 until sideN; j <- 0 until sideP)
      ptr1.put(i + j * sideN, ma(i, j))
    for (i <- 0 until sideP; j <- 0 until sideM)
      ptr2.put(i * sideM + j, mb(i, j))

    q.putWriteBuffer(buf1, true)
    q.putWriteBuffer(buf2, true)

    q.finish()
    val began = System.nanoTime()
    q.put2DRangeKernel(kernel, 0, 0, roundUpTo(sideN, tile), roundUpTo(sideM, tile), tile, tile)
    q.putReadBuffer(buf3, true)
    q.finish()

    val duration = System.nanoTime() - began
    println(f"Kernel executed in ${duration / 1000000.0}%.2fms")

    if (debug) {
      val ret = buf3.getBuffer
      val mccl = new DenseMatrix[Float](sideN, sideM)
      for (i <- 0 until sideN; j <- 0 until sideM)
        mccl.update(i, j, ret.get(i + j * sideN))

      //    val mbt = mb.t

      //    val cpuBegan = System.nanoTime()
      val mcref = ma * mb
      //    val cpuDuration = System.nanoTime() - cpuBegan
      //    println(f"CPU naïve algorithm executed in ${cpuDuration / 1000000.0}%.2fms")
      //    mcref.hashCode()

      println(s"----- C (CL) -----\n\n$mccl\n\n")
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
    ((v - 1) / modulo + 1) * modulo
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
