package com.github.hajile.symbola.cl

import com.google.common.base.Charsets
import com.google.common.io.Resources
import com.jogamp.opencl.CLDevice.Type
import com.jogamp.opencl.CLMemory.Mem
import com.jogamp.opencl.{CLBuffer, CLContext, CLPlatform}
import java.nio.FloatBuffer
import scala.util.Random
import com.jogamp.opencl.util.Filter

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

  val side = 1024
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
  kernel.setArg(2, 64)
  kernel.setArg(3, 64)
  kernel.setArg(4, 64)
  kernel.setArg(5, buf3)

  val rng = new Random

  for (i <- 0 until 10) {
    val ptr1 = buf1.getBuffer
    val ptr2 = buf2.getBuffer
    for (i <- 0 until ptr1.limit())
      ptr1.put(i, rng.nextFloat())
    for (i <- 0 until ptr2.limit())
      ptr2.put(i, rng.nextFloat())

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
    for (j <- 0 until ret.limit()) {
      val x = j % side
      val y = j - x
      require(ret.get(x + y*side) == ptr1.get(x + y*side) * ptr2.get(y + x*side))
    }
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
