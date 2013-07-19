package com.github.hajile.symbola.cl

import com.jogamp.opencl.{CLDevice, CLContext, CLPlatform}
import com.jogamp.opencl.CLDevice.Type
import com.jogamp.opencl.util.Filter

abstract class OpenCLApp extends App {
  val platformString = if (args.length >= 1) args(0).toLowerCase else ""
  val deviceString = if (args.length >= 2) args(1).toLowerCase else ""

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
  }).getMaxFlopsDevice(new Filter[CLDevice] {
    def accept(item: CLDevice) = item.getName.toLowerCase.contains(deviceString)
  })

  println(s"Using device: $device")
  val ctx = CLContext.create(device)

  val q = device.createCommandQueue()
}
