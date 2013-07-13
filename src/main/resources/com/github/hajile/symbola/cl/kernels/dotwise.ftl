<#include "common.cl">

__kernel void ${name}(<#list input as i>__global const restrict float* in_${i}, </#list>__global restrict float* out)
{
  const int i = get_global_id(0);
<#list input as i>
  const float e_${i} = in_${i}[i];
</#list>
<#list ops as o>
<#switch o.class.getSimpleName>
<#case "ScalarOp">
  const float e_${o.out} = ${o.name}(<#list o.args as i>e_${i}<#if i_has_next>, </#if></#list>);
<#break>
<#case "ConstOp">
  const float e_${o.out} = ${o.value};
<#break>
<#default>
// can't handle ${o.class.getSimpleName}
</#switch>
</#list>
  out[i] = e_${out};
}
