__kernel void ${name}(<#list input as i>__global const float* in_${i}, </#list>__global float* out)
{
  int i = get_global_id(0);
<#list input as i>
  float e_${i} = in_${i}[i];
</#list>
<#list ops as o>
  float e_${o.out} = ${o.name}(<#list o.args as i>e_${i}<#if i_has_next>, </#if></#list>);
</#list>
  out[i] = e_${out};
}
