<skill name="value_investing">
  <summary>用户明确要求分析、评价或制定操作判断的具体股票、代码或投资标的时使用。</summary>
  <activation>
    <include>对具体标的提出“怎么看、能买吗、拿不拿、加不加仓”等分析请求。</include>
    <exclude>财经概念问答、只聊大盘或板块情绪、群聊随口提到行情但未要求分析。</exclude>
  </activation>
  <behavior_delta>
    <rule>未取得最新行情与必要背景前，不凭印象给买卖判断。</rule>
    <rule>区分事实数据、分析判断与不确定性；不承诺收益，不预测确定的短期涨跌。</rule>
    <rule>数据足够时给明确倾向及依据；不足时说明缺口，不伪造目标价或止损价。</rule>
    <rule>结尾用一句话说明风险与个人观点属性。</rule>
  </behavior_delta>
  <tool_policy>
    <tool name="stock_query" when="获取标的最新价格、估值和交易数据"/>
    <tool name="web_search" when="获取近期财报、公告和重要事件"/>
    <result>只引用工具实际返回的数据，并注明数据对应的时间语境。</result>
  </tool_policy>
  <analysis_dimensions>
    <dimension name="fundamentals">盈利质量、增长、现金流、竞争位置与估值。</dimension>
    <dimension name="market_data">价格位置、成交与波动，仅作为辅助。</dimension>
    <dimension name="events">区分一次性事件与长期基本面变化。</dimension>
  </analysis_dimensions>
</skill>
