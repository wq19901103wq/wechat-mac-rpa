<skill name="tuya_smart_home">
  <summary>用户明确要求查询、开启、关闭或调节智能家居设备，或执行智能家居场景时使用。</summary>
  <activation>
    <include>消息表达了对真实智能设备状态或动作的明确目标，而非泛泛讨论设备。</include>
    <exclude>只咨询智能家居知识、描述设备但没有查询或控制请求、比喻和玩笑。</exclude>
  </activation>
  <behavior_delta>
    <rule>从完整语义识别目标设备、动作、位置和参数，不按单个动词或设备词触发。</rule>
    <rule>设备指代唯一时直接执行；候选不唯一时只询问必要的消歧信息。</rule>
    <rule>不猜设备名称、房间或工具未返回的状态。</rule>
    <limit device="air_conditioner" temperature="16-30C"/>
    <limit device="floor_heating" temperature="18-28C"/>
    <rule>超出工具能力或安全范围的删除、重置、配置修改不执行。</rule>
  </behavior_delta>
  <tool_policy>
    <tool name="tuya_list_devices" when="查询可用设备或需要消歧"/>
    <tool name="tuya_control_device" when="明确开关某个唯一设备"/>
    <tool name="tuya_set_temperature" when="明确设置唯一温控设备及合法温度"/>
    <result>按工具实际结果反馈设备、动作和状态；失败时反馈原因。</result>
  </tool_policy>
</skill>
