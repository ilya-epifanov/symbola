package com.github.hajile.symbola.cl.kernels

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers

class ProgramTemplateSpec extends FlatSpec with ShouldMatchers with ExtractKernel {
  "kernel template" should "render simple dot-wise function template" in {
    val result = ProgramTemplate("dotwise", Map(
      "name" -> "ew_sin",
      "input" -> Seq("in"),
      "ops" -> Seq(Map("out" -> "1", "name" -> "sin", "args" -> Seq("in"))),
      "out" -> "1"
    ))

    result.extractKernel() should equal(
      """
        |__kernel void ew_sin(__global const restrict float* in_in, __global restrict float* out)
        |{
        |  const int i = get_global_id(0);
        |  const float e_in = in_in[i];
        |  const float e_1 = sin(e_in);
        |  out[i] = e_1;
        |}""".stripMargin.extractKernel())
  }
}
