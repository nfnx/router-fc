# Router 云函数接口文档

## 1. 服务用途

Router 接收原始用户任务，使用 GPU 上的 SKILLRET-Embedding-0.6B 从 734 个技能中检索前 5 个候选，再向配置的 DeepSeek 复核模型发送一次请求，判断任务类型、技能适配性、速度偏好与执行模型约束。程序依据仓库既有评测策略选择 harness 与 model。

**这是任务路由服务。它返回路由建议，不执行技能、启动 Agent、调用所选执行模型完成任务，也不解析图片或 PDF。** 文档解析由独立 OXR 服务提供。

业务接口只有 **1 个：`POST /route`**。另外有根信息、存活、就绪与 OpenAPI 辅助接口。734 个技能全部参与检索；每次最多返回 1 个技能 ID，也允许空列表。技能 ID 是注册表中的稳定 ID，不能当作文件路径或技能名称。

## 2. 地址与鉴权

Base URL：

```text
https://rethos-router-rpboglliga.cn-hangzhou.fcapp.run
```

所有外部请求都必须携带独立的 Router FC 客户端 Token：

```http
Authorization: Bearer <FC_TOKEN>
```

Token 在本机受保护 `.env.fc-test` 的 `ROUTER_FC_TOKEN` 中；预填 Token 的本地交付为项目根目录 `接口文档.private.html`，不要提交或共享该文件。无需给调用者阿里云 AccessKey 或复核模型凭据。OXR 的 Token 与 Router 的 Token 不通用。

## 3. 业务请求：提交任务路由

```http
POST https://rethos-router-rpboglliga.cn-hangzhou.fcapp.run/route
Authorization: Bearer <FC_TOKEN>
Content-Type: application/json

{
  "request": "分析上传的实验 CSV，使用 Python 做回归分析、置信区间和残差图。"
}
```

`request` 是唯一允许的字段，必须是非空字符串，最多 32000 个字符。整个请求体最大 1 MiB。直接传原始任务文本；此接口不接受文件、文件 URL、历史消息、模型选择字段或已经提取的标签。速度或执行模型要求写在任务文字里，例如“优先低延迟，只使用 GLM”。

检索模型的表示长度最多 4096 token（含查询前缀），超长文本向右截断；复核模型收到原始任务文字。候选技能正文按最多 50000 字符总预算裁剪，因此技能适配判断也受候选片段覆盖范围影响。重要约束宜写在任务开头。

成功返回 HTTP 200，且只有以下三个字段。下例仅表示响应结构，不保证每次选择相同组合：

```json
{
  "skills": [],
  "harness": "claude",
  "model": "deepseek"
}
```

| 字段 | 含义 | 范围 |
| --- | --- | --- |
| skills | 适用技能的稳定 ID | 0 或 1 个 ID；来自技能注册表 |
| harness | 建议的执行框架 | codex、claude、dsh |
| model | 建议的执行模型提供方 | deepseek、glm |

云端实测中，“使用 GLM 分析实验数据”返回过以下结果，其中 ID 对应 `exploratory-data-analysis`：

```json
{
  "skills": ["491075ab-c59a-5964-b04f-60e07fcf2e98"],
  "harness": "claude",
  "model": "glm"
}
```

同一请求的技能选择可能受候选与复核模型影响，不应硬编码这个示例结果。

支持的评测任务类型为固定语料检索、深度研究、研究报告、科学分析执行；混合任务依据明确主任务选择。缺少覆盖证据或其他任务采用配置中的默认组合。空技能是正常结果。评测策略是历史证据，不是未来任务质量保证。复核模型始终使用部署配置；返回 `model: glm` 不表示本次复核请求改用了 GLM。

### curl

在终端设置 `ROUTER_FC_TOKEN` 环境变量后：

```bash
curl --connect-timeout 15 --max-time 900 \
  'https://rethos-router-rpboglliga.cn-hangzhou.fcapp.run/route' \
  -H "Authorization: Bearer ${ROUTER_FC_TOKEN}" \
  -H 'Content-Type: application/json' \
  --data '{"request":"调查钠离子电池衰减的最新研究，比较证据与局限，给出来源。"}'
```

### Python

```python
import os
import requests

response = requests.post(
    "https://rethos-router-rpboglliga.cn-hangzhou.fcapp.run/route",
    headers={"Authorization": "Bearer " + os.environ["ROUTER_FC_TOKEN"]},
    json={"request": "只使用 GLM，分析实验 CSV 并生成统计图。"},
    timeout=(15, 900),
)
response.raise_for_status()
route = response.json()
print(route)
```

### Postman

导入 `deploy/fc/router-fc.postman_collection.json`；`base_url` 已填写，`fc_token` 填 Router 客户端 Token。在 Authorization 选择继承集合鉴权；Body → raw → JSON，填写上面的 `request` 对象。首次冷启动将客户端超时调到 900000 ms。集合同时包含正常路由、GLM/DeepSeek 约束与错误输入断言；错误用例预期 4xx。

## 4. 辅助接口（逐项完整请求）

### 根信息

```http
GET https://rethos-router-rpboglliga.cn-hangzhou.fcapp.run/
Authorization: Bearer <FC_TOKEN>
```

返回 `{"service":"rethos-router-fc","version":"1.0.0","route_path":"/route"}`。

### 存活

```http
GET https://rethos-router-rpboglliga.cn-hangzhou.fcapp.run/health/live
Authorization: Bearer <FC_TOKEN>
```

返回 `{"status":"alive"}`。这只代表 HTTP 进程存活，不证明上游模型可用。

### 就绪

```http
GET https://rethos-router-rpboglliga.cn-hangzhou.fcapp.run/health/ready
Authorization: Bearer <FC_TOKEN>
```

返回 `{"status":"ready"}` 代表技能、模型和索引初始化完成；尚未初始化或初始化失败返回 503。不会为每次健康检查发送复核模型请求。

### OpenAPI

```http
GET https://rethos-router-rpboglliga.cn-hangzhou.fcapp.run/openapi.json
Authorization: Bearer <FC_TOKEN>
```

接口定义也保存在 `deploy/fc/openapi.json`。FastAPI 默认交互页 `/docs`、`/redoc` 同样受 FC 鉴权；普通浏览器地址栏不会自动携带 Token，建议用 Postman 或本地导入 OpenAPI。

`POST /initialize` 是 FC 生命周期回调，不是日常业务接口，不应由客户端反复调用。

## 5. 错误与重试

| HTTP | 来源与含义 | 处理 |
| --- | --- | --- |
| 400 | 空白、超长、错误类型、额外字段、重复 JSON 键或无效 JSON | 修正请求；不调用复核模型 |
| 403 | FC 客户端 Token 缺失或错误 | 检查触发器中启用的 Token |
| 404 / 405 | 路径或方法错误 | 使用 POST /route |
| 413 | 请求体超过 1 MiB | 缩小文本 |
| 415 | 不是 application/json | 设置 Content-Type |
| 429 | 实例忙或 FC 平台限流 | 有界退避，避免持续重复提交 |
| 502 | 复核模型网络、超时或不合法决策 | 记录失败；服务不会自动补第二次模型调用或伪造成功 |
| 503 | Router 未就绪 | 查看初始化日志，再有限重试 |
| 500 | 初始化或路由内部失败 | 管理员检查 FC 日志 |

FC 平台还可能返回 `ResourceExhausted`、`ResourceThrottled`、初始化失败等 JSON，其结构不同于 FastAPI 的 `detail` 错误。GPU 配额足够不代表随时有库存。

本接口没有幂等键，也不保存任务结果。遇到连接中断时不能判定服务器是否已完成复核；重试可能再次产生模型费用。不要自动无限重试。

## 6. 部署与运行边界

函数名 `rethos-router`，杭州 Ampere 24 GB GPU，8 vCPU、32 GB 内存，单实例并发 1，最小实例数 0。无可用实例时由 FC 分配 GPU，初始化后处理请求；已有空闲实例可复用，不是每次调用都新开一台机器。空闲回收时间由平台决定，不承诺固定保留时长。

镜像保存代码和依赖；权重、734 个技能与检索索引保存在独立 NAS 子目录 `/router`，函数挂载到 `/mnt/router-assets`。实例回收不会删除这些文件。新实例仍需读取权重并加载 GPU；首次无缓存索引时还需构建技能向量。索引身份包含技能顺序、正文摘要、权重和依赖版本，变化后自动重建。NAS 不代表零冷启动时间。

权重固定为 `ThakiCloud/SKILLRET-Embedding-0.6B` 的 revision `67f1594c46c40e13a5eb2f4092cc85c98c9ed32d`，主权重 1191586416 字节，SHA256 `f73118cac018ffa7ebb5a1ffbdf82034490dfb7f2559558f1e79277f1e8de172`。运行期间仅从 NAS 本地路径读取，不在线下载权重。

FC 执行超时 240 秒，初始化回调上限 660 秒，内部初始化上限 600 秒；复核上游请求超时 120 秒。客户端首次调用需为镜像准备与冷启动预留时间。不要根据一次短测承诺吞吐或 SLA。

生产调用可另配自定义域名；当前交付使用现有 FC 默认域名。详细实际验证及未验证边界见 `docs/FC部署验收.md`。

本轮首次就绪实测约 196 秒，包含首次初始化；修正版在已存在 NAS 索引时就绪约 7.06 秒，前面另发生过镜像加速未就绪的 412，不能把 7.06 秒当作从发布镜像到可用的总时长。真实业务请求约 2.33–13.64 秒，受外部复核模型延迟影响；短测不代表长期时延保证。
