package com.github.hajile.symbola.cl

import com.nativelibs4java.opencl.JavaCL
import com.nativelibs4java.opencl.CLMem.Usage
import java.nio.ByteBuffer
import scala.util.Random
import scala.io.Source
import java.nio.file.Files
import com.nativelibs4java.util.IOUtils
import org.bridj.Pointer

class OpenCLExample

object OpenCLExample extends App {
  for (p <- JavaCL.listPlatforms()) {
    println("Profile: " + p.getName)
    for (d <- p.listAllDevices(false)) {
      println("  Device: " + d.getName)
    }
  }

  val ctx = JavaCL.createBestContext()
  val q = ctx.createDefaultQueue()

  val vectorSize = 1048576

  val buf1 = ctx.createFloatBuffer(Usage.Input, vectorSize)
  val buf2 = ctx.createFloatBuffer(Usage.Input, vectorSize)
  val buf3 = ctx.createFloatBuffer(Usage.Output, vectorSize)

  val src = IOUtils.readText(classOf[OpenCLExample].getResource("kernels/kernel1.cl"))
  val program = ctx.createProgram(src)

//  program.setFastRelaxedMath()
//  program.setMadEnable()
//  program.setUnsafeMathOptimizations()

  program.build()

  val kernel = program.createKernel("kernel2", buf1, buf2, buf3)

  val rng = new Random

  for (i <- 0 until 10) {
    val ptr1 = Pointer.allocateFloats(vectorSize)
    val ptr2 = Pointer.allocateFloats(vectorSize)
    for (i <- 0 until ptr1.getValidElements.toInt)
      ptr1.set(i, rng.nextGaussian().toFloat)
    for (i <- 0 until ptr2.getValidElements.toInt)
      ptr2.set(i, rng.nextGaussian().toFloat)

    buf1.write(q, ptr1, false)
    buf2.write(q, ptr2, false)

    q.finish()
    val began = System.nanoTime()
    val kernelCompletion = kernel.enqueueNDRange(q, Array(vectorSize), Array(1))
    kernelCompletion.waitFor()
    val ret = buf3.read(q)

    val duration = System.nanoTime() - began
    println(f"Kernel executed in ${duration/1000000.0}%.2fms")

//    for (j <- 0 until ret.getValidElements.toInt) {
//      require(ret.get(j) == ptr1.get(j) * ptr2.get(j))
//    }

    ptr1.release()
    ptr2.release()
    ret.release()
  }

  def dumpBuffer(ptr: Pointer[Float]): String = {
    val str = StringBuilder.newBuilder
    for (i <- 0 until ptr.getValidElements.toInt) {
      if (i != 0)
        str.append(' ')
      str.append(ptr.get(i))
    }
    str.toString()
  }
}
