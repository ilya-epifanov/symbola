package com.github.hajile.symbola.cl

import com.google.common.base.Charsets
import com.google.common.io.Resources
import com.jogamp.opencl.CLDevice.Type
import com.jogamp.opencl.CLMemory.Mem
import com.jogamp.opencl.{CLBuffer, CLContext, CLPlatform}
import java.nio.FloatBuffer
import scala.util.Random
import com.jogamp.opencl.util.Filter
import breeze.linalg.DenseMatrix

class OpenCLExample

object OpenCLExample extends App {
  val platformString = if (args.length >= 1) args(0).toLowerCase else ""

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

  val side = 16
  val vectorSize = side * side

  val buf1 = ctx.createFloatBuffer(vectorSize, Mem.ALLOCATE_BUFFER)
  val buf2 = ctx.createFloatBuffer(vectorSize, Mem.ALLOCATE_BUFFER)
  val buf3 = ctx.createFloatBuffer(vectorSize, Mem.ALLOCATE_BUFFER)

  val src = Resources.toString(classOf[OpenCLExample].getResource("kernels/kernels.cl"), Charsets.UTF_8)
  val program = ctx.createProgram(src)

  //  program.setFastRelaxedMath()
  //  program.setMadEnable()
  //  program.setUnsafeMathOptimizations()

  program.build()

  val kernel = program.createCLKernel("mmultopt")
  kernel.setArg(0, buf1)
  kernel.setArg(1, buf2)
  kernel.setArg(2, side / 16)
  kernel.setArg(3, side / 16)
  kernel.setArg(4, side / 16)
  kernel.setArg(5, buf3)

  val rng = new Random
  val ma = new DenseMatrix[Float](side, side)
  for (i <- 0 until side; j <- 0 until side)
    ma.update(i, j, rng.nextFloat())

  val mb = new DenseMatrix[Float](side, side)
  for (i <- 0 until side; j <- 0 until side)
    mb.update(i, j, rng.nextFloat())

  println(s"----- A -----\n\n$ma\n\n")
  println(s"----- B -----\n\n$mb\n\n")

  for (i <- 0 until 1) {
    val ptr1 = buf1.getBuffer
    val ptr2 = buf2.getBuffer
    for (i <- 0 until side; j <- 0 until side)
      ptr1.put(i + j*side, ma(i, j))
    for (i <- 0 until side; j <- 0 until side)
      ptr2.put(i + j*side, mb(i, j))

    q.putWriteBuffer(buf1, true)
    q.putWriteBuffer(buf2, true)

    q.finish()
    val began = System.nanoTime()
    q.put2DRangeKernel(kernel, 0, 0, side, side, 16, 16)
    q.putReadBuffer(buf3, true)
    q.finish()

    val duration = System.nanoTime() - began
    println(f"Kernel executed in ${duration / 1000000.0}%.2fms")

    val ret = buf3.getBuffer
    val mccl = new DenseMatrix[Float](side, side)
    for (i <- 0 until side; j <- 0 until side)
      mccl.update(i, j, ptr2.get(i + j*side))

    val mcref = ma * mb.t

    println(s"----- C (CL) -----\n\n$mccl\n\n")
    println(s"----- C (ref) -----\n\n$mcref\n\n")

//    for (j <- 0 until ret.limit()) {
//      val x = j % side
//      val y = j / side
//      val r = ret.get(x + y * side)
//      println(s"[$x][$y] : $r")
//    }
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
