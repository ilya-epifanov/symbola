package com.github.hajile.symbola.cl.kernels

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers

class DotwiseSpec extends FlatSpec with ShouldMatchers with ExtractKernel {
  "dotwise template" should "render a no-op" in {
    val result = ProgramTemplate("dotwise", DotwiseKernel("ew_identity", "a", Seq("a"), Seq()))

    result.extractKernel() should equal(
      """
        |__kernel void ew_identity(__global const restrict float* in_a, __global restrict float* out)
        |{
        |  const int i = get_global_id(0);
        |  const float e_a = in_a[i];
        |  out[i] = e_a;
        |}""".stripMargin.extractKernel())
  }

  it should "render double application" in {
    val result = ProgramTemplate("dotwise", DotwiseKernel("ew_identity", "2", Seq("a"), Seq(
      ScalarOp("sin", "1", "a"), ScalarOp("sqrt", "2", "1"))))

    result.extractKernel() should equal(
      """
        |__kernel void ew_identity(__global const restrict float* in_a, __global restrict float* out)
        |{
        |  const int i = get_global_id(0);
        |  const float e_a = in_a[i];
        |  const float e_1 = sin(e_a);
        |  const float e_2 = sqrt(e_1);
        |  out[i] = e_2;
        |}""".stripMargin.extractKernel())
  }

  it should "render binary operation" in {
    val result = ProgramTemplate("dotwise", DotwiseKernel("dotwise_product", "1", Seq("a", "b"),
      Seq(ScalarOp("mul", "1", "a", "b"))))

    result.extractKernel() should equal(
      """
        |__kernel void dotwise_product(__global const restrict float* in_a, __global const restrict float* in_b, __global restrict float* out)
        |{
        |  const int i = get_global_id(0);
        |  const float e_a = in_a[i];
        |  const float e_b = in_b[i];
        |  const float e_1 = mul(e_a, e_b);
        |  out[i] = e_1;
        |}""".stripMargin.extractKernel())
  }

  it should "render constant expressions" in {
    val result = ProgramTemplate("dotwise", DotwiseKernel("recip", "2", Seq("a"),
      Seq(ConstOp("1", "1.0f"), ScalarOp("div", "2", "1", "a"))))

    result.extractKernel() should equal(
      """
        |__kernel void recip(__global const restrict float* in_a, __global restrict float* out)
        |{
        |  const int i = get_global_id(0);
        |  const float e_a = in_a[i];
        |  const float e_1 = 1.0f;
        |  const float e_2 = div(e_1, e_a);
        |  out[i] = e_2;
        |}""".stripMargin.extractKernel())
  }
}
