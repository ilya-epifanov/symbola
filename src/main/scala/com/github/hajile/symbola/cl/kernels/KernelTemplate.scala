package com.github.hajile.symbola.cl.kernels

import freemarker.template.{Version, TemplateExceptionHandler, Configuration}
import java.io.StringWriter

object KernelTemplate {
  private val cfg = new Configuration()
  cfg.setClassForTemplateLoading(getClass, "")
  cfg.setDefaultEncoding("UTF-8")
  cfg.setTemplateExceptionHandler(TemplateExceptionHandler.DEBUG_HANDLER)
  cfg.setIncompatibleImprovements(new Version("2.3.20"))
  cfg.setObjectWrapper(new ScalaObjectWrapper)

  def apply(name: String, args: Map[String, Any]): String = {
    val template = cfg.getTemplate(s"$name.ftl")
    val ret = new StringWriter()
    template.process(args, ret)
    ret.toString
  }
}