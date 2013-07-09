package com.github.hajile.symbola.cl

import com.google.common.base.Charsets
import com.google.common.io.Resources
import com.jogamp.opencl.CLDevice.Type
import com.jogamp.opencl.CLMemory.Mem
import com.jogamp.opencl.{CLBuffer, CLContext, CLPlatform}
import java.nio.FloatBuffer
import scala.util.Random

class OpenCLExample

object OpenCLExample extends App {
  for (p <- CLPlatform.listCLPlatforms()) {
    println("Platform: " + p.getName)
    for (d <- p.listCLDevices(Type.ALL)) {
      println("  Device: " + d.getName)
    }
  }

  val ctx = CLContext.create()
  val device = ctx.getMaxFlopsDevice(Type.GPU)
  println(s"Using device: $device")

  val q = device.createCommandQueue()

  val vectorSize = 1048576

  val buf1 = ctx.createFloatBuffer(vectorSize, Mem.ALLOCATE_BUFFER)
  val buf2 = ctx.createFloatBuffer(vectorSize, Mem.ALLOCATE_BUFFER)
  val buf3 = ctx.createFloatBuffer(vectorSize, Mem.ALLOCATE_BUFFER)

  val src = Resources.toString(classOf[OpenCLExample].getResource("kernels/kernel1.cl"), Charsets.UTF_8)
  val program = ctx.createProgram(src)

  //  program.setFastRelaxedMath()
  //  program.setMadEnable()
  //  program.setUnsafeMathOptimizations()

  program.build()

  val kernel = program.createCLKernel("kernel2")
  kernel.setArg(0, buf1)
  kernel.setArg(1, buf2)
  kernel.setArg(2, buf3)

  val rng = new Random

  for (i <- 0 until 10) {
    val ptr1 = buf1.getBuffer
    val ptr2 = buf2.getBuffer
    for (i <- 0 until ptr1.remaining())
      ptr1.put(i, rng.nextGaussian().toFloat)
    for (i <- 0 until ptr2.remaining())
      ptr2.put(i, rng.nextGaussian().toFloat)

    q.putWriteBuffer(buf1, true)
    q.putWriteBuffer(buf2, true)

    q.finish()
    val began = System.nanoTime()
    q.put1DRangeKernel(kernel, 0, vectorSize, 64)
    q.putReadBuffer(buf3, true)
    q.finish()

    val duration = System.nanoTime() - began
    println(f"Kernel executed in ${duration / 1000000.0}%.2fms")

    //    for (j <- 0 until ret.getValidElements.toInt) {
    //      require(ret.get(j) == ptr1.get(j) * ptr2.get(j))
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
