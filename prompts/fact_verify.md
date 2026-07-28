逐条严格判断 evidence 是否直接支持 claims 的全部字面含义：

- `entailed`：证据直接支持 claim 的每一部分；
- `contradicted`：claim 的任一部分被证据否定；
- `unknown`：只支持一部分，或其余部分没有直接证据。

每条 claim 独立判断，不得使用其他 claim、Bot 旧回复或待检查的回复自证。不得用常识、固定搭配或“这是调侃”补足证据。

只输出 JSON：
`{"claims":[{"claim":"原 claim","verdict":"entailed/contradicted/unknown","reason":"一句话"}]}`
