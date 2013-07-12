package com.github.hajile.symbola.cl.kernels

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers

class DotwiseSpec extends FlatSpec with ShouldMatchers {
  "dotwise template" should "render a no-op" in {
    val result = KernelTemplate("elementwise", Map(
      "name" -> "ew_identity",
      "ops" -> Seq(),
      "out" -> "in"
    ))

    result.trim should equal(
      """
        |__kernel void ew_identity(__global const float* input, __global float* out)
        |{
        |  int i = get_global_id(0);
        |  float in = input[i];
        |  out[i] = in;
        |}""".stripMargin.trim)
  }

  "dotwise template" should "render double application" in {
    val result = KernelTemplate("elementwise", Map(
      "name" -> "ew_identity",
      "ops" -> Seq(
        Map("name" -> "sin", "in" -> "in", "out" -> "e1"),
        Map("name" -> "sqrt", "in" -> "e1", "out" -> "e2")
      ),
      "out" -> "e2"
    ))

    result.trim should equal(
      """
        |__kernel void ew_identity(__global const float* input, __global float* out)
        |{
        |  int i = get_global_id(0);
        |  float in = input[i];
        |  float e1 = sin(in);
        |  float e2 = sqrt(e1);
        |  out[i] = e2;
        |}""".stripMargin.trim)
  }
}
