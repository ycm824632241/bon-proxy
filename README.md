# bon-proxy

一个面向 vLLM / SGLang 的异步 Best-of-N 代理。客户端继续使用 OpenAI Chat Completions
接口；代理让回答模型一次生成 N 个候选，再由独立 judge 模型选择最佳候选。

## 行为约定

- 仅提供非流式 `POST /v1/chat/completions`；`stream=true` 返回 400。
- 候选数量由 YAML 中的 `answer.params.n` 固定。客户端传入的 `n` 被忽略，最终始终
  返回一个 `choice`。
- 客户端的 `model`、`temperature`、`top_p`、`n` 和 `chat_template_kwargs` 被 YAML
  覆盖，其他请求字段（包括 tools、response_format 和 vLLM 扩展字段）透传到回答端。
- 回答模型和 judge 模型必须暴露 OpenAI-compatible `/v1/chat/completions`。
- `answer.return_token_ids=true` 时，回答端必须返回每个 choice 的 token IDs，代理据此
  重算获胜回答的 `usage.completion_tokens`。SGLang v0.5.16 应配置为 `false`；此时代理
  保留 SGLang 返回的全部 N 个候选的聚合 usage。
- judge 只返回候选索引；客户端收到的是回答模型返回的对应原始 choice（包括原始
  `index`、`message`、`finish_reason` 等字段），代理不会改写 choice，也不会把 judge
  的文字、评分或说明拼接到候选内容中。
- judge 失败或超时时降级到第一个候选；回答模型超时返回 504。
- 入口不校验 Bearer key；上游 key 从 YAML 读取。

## 安装与启动

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp config.example.yaml config.yaml
bon-proxy --config config.yaml
```

SGLang v0.5.16 示例：

```yaml
answer:
  base_url: http://sglang:30001/v1
  api_key: ""
  model: served-model-name
  timeout_seconds: 1800
  return_token_ids: false
  params:
    temperature: 1.0
    top_p: 0.95
    n: 4
    chat_template_kwargs: {}
```

所有配置都位于 YAML。服务固定使用单个 Uvicorn worker，因此 `max_concurrency` 是当前
proxy 实例的精确工作流并发上限；多副本的总并发是所有副本上限之和。

## openai-python 调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="virtual-model",
    messages=[{"role": "user", "content": "写一个快速排序"}],
    temperature=0.2,  # 被 YAML 的回答模型参数覆盖
    n=8,  # 被忽略；候选数由 answer.params.n 决定
)
print(response.choices[0].message)
```

## 测试

```bash
pytest
ruff check .
```

测试使用模拟的回答和 judge vLLM，不会连接真实模型或 API key。正式联调时在
`config.yaml` 中填入服务器上的两个 vLLM 地址即可。
