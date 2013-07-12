package com.github.hajile.symbola.cl.kernels

import com.google.common.base.Charsets
import com.google.common.io.Resources
import org.fusesource.scalate.TemplateEngine
import org.fusesource.scalate.util.{ResourceLoader, Resource}

object KernelTemplate {
  private val engine = new TemplateEngine
  engine.resourceLoader = new ResourceLoader {
    override def resource(uri: String): Option[Resource] = {
      val url = getClass.getResource(uri)

      Some(Resource.fromText(uri, Resources.toString(url, Charsets.US_ASCII)))
    }
  }

  def apply(name: String, args: Map[String, Any]): String = {
    engine.layout(s"$name.mustache", args)
  }
}