package com.github.hajile.symbola.cl.kernels

import freemarker.template.{TemplateMethodModelEx, TemplateHashModelEx, TemplateHashModel, TemplateScalarModel, TemplateModelIterator, TemplateCollectionModel, TemplateSequenceModel, SimpleScalar, SimpleNumber, TemplateBooleanModel, TemplateDateModel, TemplateModel, ObjectWrapper}
import java.lang.reflect.{Field, Modifier, InvocationTargetException, Method}
import java.util.Date
import scala.annotation.tailrec

class ScalaObjectWrapper extends ObjectWrapper {

  override def wrap(obj: Any): TemplateModel = obj match {
    // Basic types
    case null => null
    case option: Option[Any] => option match {
      case Some(o) => wrap(o)
      case _ => null
    }
    case model: TemplateModel => model
    // Scala base types
    case seq: Seq[Any] => new ScalaSeqWrapper(seq, this)
    case array: Array[Any] => new ScalaArrayWrapper(array, this)
    case map: Map[_, _] => new ScalaMapWrapper(map.asInstanceOf[Map[Any, Any]], this)
    case it: Iterable[Any] => new ScalaIterableWrapper(it, this)
    case it: Iterator[Any] => new ScalaIteratorWrapper(it, this)
    case str: String => new SimpleScalar(str)
    case date: Date => new ScalaDateWrapper(date, this)
    case num: Number => new SimpleNumber(num)
    case bool: Boolean =>
      if (bool) TemplateBooleanModel.TRUE else TemplateBooleanModel.FALSE
    // Everything else
    case o => new ScalaBaseWrapper(o, this)
  }

}

class ScalaDateWrapper(val date: Date, wrapper: ObjectWrapper)
        extends TemplateDateModel {

  def getDateType = TemplateDateModel.UNKNOWN

  def getAsDate = date

}

class ScalaSeqWrapper[T](val seq: Seq[T], wrapper: ObjectWrapper)
        extends TemplateSequenceModel {

  def get(index: Int) = wrapper.wrap(seq(index))

  def size = seq.size

}

class ScalaArrayWrapper[T](val array: Array[T], wrapper: ObjectWrapper)
        extends TemplateSequenceModel {

  def get(index: Int) = wrapper.wrap(array(index))

  def size = array.length

}

class ScalaMapWrapper(val map: Map[Any, Any], wrapper: ObjectWrapper)
        extends TemplateHashModelEx {

  override def get(key: String) = wrapper.wrap(map.get(key))

  override def isEmpty = map.isEmpty

  def values = new ScalaIterableWrapper(map.values, wrapper)

  val keys = new ScalaIterableWrapper(map.keys, wrapper)

  def size = map.size

}

class ScalaIterableWrapper[T](val it: Iterable[T], wrapper: ObjectWrapper)
        extends TemplateCollectionModel {

  def iterator = new ScalaIteratorWrapper(it.iterator, wrapper)

}

class ScalaIteratorWrapper[T](val it: Iterator[T], wrapper: ObjectWrapper)
        extends TemplateModelIterator with TemplateCollectionModel {

  def next = wrapper.wrap(it.next())

  def hasNext = it.hasNext

  def iterator = this

}

class ScalaMethodWrapper(val target: Any,
                         val methodName: String,
                         val wrapper: ObjectWrapper)
        extends TemplateMethodModelEx {

  def exec(arguments: java.util.List[_]) = {
    val args = arguments.toArray
    val result = try {
      val method = target.getClass.getMethod(methodName, args.map(_.getClass): _*)
      method.invoke(target, args)
    } catch {
      case e: InvocationTargetException if e.getCause != null =>
        throw e.getCause
    }
    wrapper.wrap(result)
  }
}

class ScalaBaseWrapper(val obj: Any, val wrapper: ObjectWrapper)
        extends TemplateHashModel with TemplateScalarModel {

  val objectClass = obj.asInstanceOf[Object].getClass

  @tailrec
  private def findMethod(cl: Class[_], name: String): Option[Method] =
    cl.getMethods.toList.find {
      m =>
        m.getName.equals(name) && Modifier.isPublic(m.getModifiers)
    } match {
      case None if cl != classOf[Object] =>
        findMethod(cl.getSuperclass, name)
      case other => other
    }

  @tailrec
  private def findField(cl: Class[_], name: String): Option[Field] =
    cl.getFields.toList.find {
      f =>
        f.getName.equals(name) && Modifier.isPublic(f.getModifiers)
    } match {
      case None if cl != classOf[Object] => findField(cl.getSuperclass, name)
      case other => other
    }

  def get(key: String): TemplateModel = {
    val o = obj.asInstanceOf[Object]
    if (key.startsWith("$"))
      return wrapper.wrap(null)
    if (key == "class")
      return wrapper.wrap(o.getClass)
    findField(objectClass, key) match {
      case Some(field) => return wrapper.wrap(field.get(o))
      case _ =>
    }
    findMethod(objectClass, key) match {
      case Some(method) if method.getParameterTypes.length == 0 =>
        return wrapper.wrap(method.invoke(obj))
      case Some(method) =>
        return new ScalaMethodWrapper(obj, method.getName, wrapper)
      case _ =>
    }
    // nothing found
    ObjectWrapper.DEFAULT_WRAPPER.wrap(obj)
  }

  def isEmpty = false

  def getAsString = obj.toString
}
