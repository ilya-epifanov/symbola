package com.github.hajile.symbola.cl.kernels

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers

class KernelTemplateSpec extends FlatSpec with ShouldMatchers {
  "kernel template" should "render simple dot-wise function template" in {
    val result = KernelTemplate("dotwise", Map(
      "name" -> "ew_sin",
      "input" -> Seq("in"),
      "ops" -> Seq(Map("out" -> "1", "name" -> "sin", "args" -> Seq("in"))),
      "out" -> "1"
    ))

    result.trim should equal(
      """
        |__kernel void ew_sin(__global const float* in_in, __global float* out)
        |{
        |  int i = get_global_id(0);
        |  float e_in = in_in[i];
        |  float e_1 = sin(e_in);
        |  out[i] = e_1;
        |}""".stripMargin.trim)
  }
}
