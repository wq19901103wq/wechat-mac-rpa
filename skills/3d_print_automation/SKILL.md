<skill name="3d_print_automation">
  <summary>用户明确要读取或修改 3MF、缩放模型、调整打印支撑，或查询 3D 打印机状态时使用。</summary>
  <activation>
    <include>请求的目标是 3MF 文件、模型缩放、支撑参数或打印机状态。</include>
    <exclude>只讨论 3D 打印知识，没有要求读取、修改或查询。</exclude>
  </activation>
  <behavior_delta>
    <rule>缺少完成操作所必需的文件路径或打印机连接参数时，只询问缺失项。</rule>
    <rule>不猜文件路径、输出路径或打印机连接信息。</rule>
    <rule>参数越界时不执行，并给出允许范围。</rule>
    <limits>
      <parameter name="branch_diameter" range="2-15mm" default="3mm"/>
      <parameter name="threshold_angle" range="30-60deg" default="45deg"/>
      <parameter name="wall_count" range="0-4" default="1"/>
      <parameter name="interface_layers" range="2-6" default="3"/>
    </limits>
    <default_output operation="scale">原文件名_scaled.3mf</default_output>
    <default_output operation="support">原文件名_supported.3mf</default_output>
  </behavior_delta>
  <tool_policy>
    <tool name="print3d_read_3mf" when="读取或检查 3MF" required="file_path"/>
    <tool name="print3d_scale_model" when="缩放模型" required="input_file,scale,output_file"/>
    <tool name="print3d_update_support" when="修改支撑" required="input_file,output_file"/>
    <tool name="print3d_get_printer_status" when="查询打印机状态" required="ip,access_code,serial"/>
    <result>成功时反馈实际文件路径、尺寸、关键参数或状态；失败时反馈工具返回的原因。</result>
  </tool_policy>
</skill>
