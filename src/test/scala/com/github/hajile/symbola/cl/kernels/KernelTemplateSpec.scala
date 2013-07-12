package com.github.hajile.symbola.cl.kernels

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers

class KernelTemplateSpec extends FlatSpec with ShouldMatchers {
  "kernel template" should "render simple dot-wise function template" in {
    val result = KernelTemplate("elementwise", Map(
      "name" -> "ew_sin",
      "ops" -> Seq(Map("out" -> "e1", "name" -> "sin", "in" -> "in")),
      "out" -> "e1"
    ))

    result.trim should equal(
      """
        |__kernel void ew_sin(__global const float* input, __global float* out)
        |{
        |  int i = get_global_id(0);
        |  float in = input[i];
        |  float e1 = sin(in);
        |  out[i] = e1;
        |}""".stripMargin.trim)
  }
}
