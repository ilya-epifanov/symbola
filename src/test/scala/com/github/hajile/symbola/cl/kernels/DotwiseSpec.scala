package com.github.hajile.symbola.cl.kernels

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers

class DotwiseSpec extends FlatSpec with ShouldMatchers {
  "dotwise template" should "render a no-op" in {
    val result = KernelTemplate("dotwise", Map(
      "name" -> "ew_identity",
      "input" -> Seq("a"),
      "ops" -> Seq(),
      "out" -> "a"
    ))

    result.trim should equal(
      """
        |__kernel void ew_identity(__global const float* in_a, __global float* out)
        |{
        |  int i = get_global_id(0);
        |  float e_a = in_a[i];
        |  out[i] = e_a;
        |}""".stripMargin.trim)
  }

  it should "render double application" in {
    val result = KernelTemplate("dotwise", Map(
      "name" -> "ew_identity",
      "input" -> Seq("a"),
      "ops" -> Seq(
        Map("name" -> "sin", "args" -> Seq("a"), "out" -> "1"),
        Map("name" -> "sqrt", "args" -> Seq("1"), "out" -> "2")
      ),
      "out" -> "2"
    ))

    result.trim should equal(
      """
        |__kernel void ew_identity(__global const float* in_a, __global float* out)
        |{
        |  int i = get_global_id(0);
        |  float e_a = in_a[i];
        |  float e_1 = sin(e_a);
        |  float e_2 = sqrt(e_1);
        |  out[i] = e_2;
        |}""".stripMargin.trim)
  }

  it should "render binary operation" in {
    val result = KernelTemplate("dotwise", Map(
      "name" -> "dotwise_product",
      "input" -> Seq("a", "b"),
      "ops" -> Seq(
        Map("name" -> "mul", "args" -> Seq("a", "b"), "out" -> "1")
      ),
      "out" -> "1"
    ))

    result.trim should equal(
      """
        |__kernel void dotwise_product(__global const float* in_a, __global const float* in_b, __global float* out)
        |{
        |  int i = get_global_id(0);
        |  float e_a = in_a[i];
        |  float e_b = in_b[i];
        |  float e_1 = mul(e_a, e_b);
        |  out[i] = e_1;
        |}""".stripMargin.trim)
  }
}
