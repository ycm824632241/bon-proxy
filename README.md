# bon-proxy

面向 vLLM / SGLang 等 OpenAI 兼容后端的异步 Best-of-N 代理。

客户端继续使用 OpenAI Chat Completions；代理先让回答模型一次生成 N 个候选，再由独立 judge 选出最佳一条返回。

| 项 | 值 |
|---|---|
| 包名 | `bon-proxy` |
| Python 模块 | `bon_proxy` |
| CLI | `bon-proxy` |
| 版本 | `0.1.0` |
| Python | ≥ 3.11 |

## 请求路径

```
客户端 (openai-python)
    → POST /v1/chat/completions
        → answer 上游 /v1/chat/completions  (n=N)
        → judge 上游 /v1/chat/completions   (n=1, json_schema)
    ← 只返回获胜的那一条
```

入口：

- `POST /v1/chat/completions`：主路径，不支持流式（`stream=true` → 400）
- `GET /v1/models`：返回 YAML 里配置的 answer 模型名
- `GET /health`

入口不校验 Bearer；上游 API key 只从 YAML 读取。

## 行为约定

- 候选数由 YAML 的 `answer.params.n` 固定（至少 2）。客户端传入的 `n` 被忽略，最终始终返回 1 个 `choice`。
- YAML 覆盖客户端的：`model`、`temperature`、`top_p`、`n`、`chat_template_kwargs`。其余字段（tools、`response_format`、vLLM / SGLang 扩展等）透传到回答端。
- 客户端收到的是回答模型返回的原始 choice（`index`、`message`、`finish_reason` 原样保留），judge 文字不会拼进结果。
- judge 只返回候选索引 `{"best_index": k}`（JSON Schema 强制）。
- judge 失败 / 超时 / 解析失败 → 降级到第 0 个候选；回答模型超时 → 504。
- `return_token_ids=true`（vLLM）：按获胜 choice 的 token IDs 重算 `completion_tokens`。
- `return_token_ids=false`（SGLang v0.5.16）：保留上游对全部 N 条的聚合 usage。
- 并发：单 Uvicorn worker + `max_concurrency`，超额不排队，直接 429。
- 多模态：messages 里的图片等非文本 part 会抽出来单独传给 judge，JSON 里只留引用。
- Judge prompt 防护：评测数据前会加 “untrusted evaluation data”，防止候选内容劫持 judge。

回答模型和 judge 模型都必须暴露 OpenAI-compatible `/v1/chat/completions`。

## 模块

| 模块 | 职责 |
|---|---|
| `app.py` | FastAPI 路由、请求校验、并发闸门 |
| `service.py` | BoN 编排：造 payload、抽候选、调 judge、拼最终响应 |
| `upstream.py` | OpenAI 兼容 `/chat/completions` 的 httpx 客户端 |
| `config.py` | YAML → Pydantic 严格校验 |
| `concurrency.py` | 无排队并发上限，满了直接 429 |
| `errors.py` | OpenAI 风格错误体 |

## 安装与启动

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp config.example.yaml config.yaml
bon-proxy --config config.yaml
```

所有配置都位于 YAML。服务固定使用单个 Uvicorn worker，因此 `max_concurrency` 是当前 proxy 实例的精确工作流并发上限；多副本的总并发是所有副本上限之和。

### SGLang 部署

仓库已附带两套示例配置和启动脚本：

```bash
./start-sglang.sh    # bon-proxy --config config.sglang.yaml
./start-sglang2.sh   # bon-proxy --config config.sglang2.yaml
```

| 文件 | 代理端口 | 说明 |
|---|---|---|
| `config.sglang.yaml` | `0.0.0.0:28081` | 与 `start-sglang.sh` 配套 |
| `config.sglang2.yaml` | `0.0.0.0:28083` | 与 `start-sglang2.sh` 配套 |

两套示例均为：

- answer / judge 共用同一 SGLang 实例
- `n=4`，`return_token_ids: false`
- 超时：answer 3600s / judge 600s
- `max_concurrency=32`

SGLang v0.5.16 片段：

```yaml
answer:
  base_url: http://sglang:30001/v1
  api_key: ""
  model: served-model-name
  timeout_seconds: 3600
  return_token_ids: false
  params:
    temperature: 1.0
    top_p: 0.95
    n: 4
    chat_template_kwargs: {}
```

## 客户端调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",  # SGLang 示例为 28081 / 28083
    api_key="unused",
)

response = client.chat.completions.create(
    model="virtual-model",
    messages=[{"role": "user", "content": "写一个快速排序"}],
    temperature=0.2,  # 被 YAML 覆盖
    n=8,              # 被忽略；候选数由 answer.params.n 决定
)
print(response.choices[0].message)
```

## 测试

```bash
pytest
ruff check .
```

测试使用模拟的回答和 judge 上游，不会连接真实模型或 API key。正式联调时在 `config.yaml` 中填入两个 OpenAI 兼容地址即可。
