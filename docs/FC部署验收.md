# Router 独立 GPU FC 部署验收

日期：2026-11-05。此文件保留首次失败及修复过程；最终业务结果以文末实测记录为准。

## 部署范围

Router 独立函数 `rethos-router`，独立 ACR 仓库 `rethos/router`、独立客户端 Token、独立 NAS 子目录 `/router`。OXR 函数 `rethos-oxr-demo` 的镜像、触发器和配置没有因本次 Router 发布而改变；本轮再次就绪检查返回 200。

业务能力为 `POST /route`：GPU SkillRet 检索前 5 候选、一次外部模型复核、程序依据原有历史评测策略选择 harness/model，响应只有 skills/harness/model。不会执行技能或完整 Agent 任务。原始 main 未覆盖；改造位于 deploy/aliyun-fc，代码推送组织原始仓库及用户个人仓库。

## 资产校验

- 用户提供的技能 ZIP：34378813 字节，SHA256 `8cb03e3f51928697ff1507387d6c75534f7f22347555d2cd4d2c19ab2e6ea8e8`。
- 本地解压校验归档路径与符号链接，734 条唯一技能记录、653 条 research_core；MANIFEST 中 3440 个文件的大小与 SHA256 全部通过。
- 技能包已上传 NAS `/router/data/research-skill-pool-v2-20261104`，云端再次校验 3440 文件与 734 条技能。
- 主权重 1191586416 字节，固定 revision `67f1594c46c40e13a5eb2f4092cc85c98c9ed32d`，SHA256 `f73118cac018ffa7ebb5a1ffbdf82034490dfb7f2559558f1e79277f1e8de172`。
- 10 个模型文件在本机按固定 Git blob/LFS 摘要校验，上传 NAS 后再次校验。模型与技能未提交 Git，也未打进镜像。
- 734 条参与检索不代表全部技能的脚本、外部 MCP、工具及供应商账号都经过执行验收；Router 只读取并选择它们。

## 失败与修复记录

1. RAM 调用 ACR 绑定 GitHub 源返回业务码 SOURCE_ACCOUNT_NOT_AVAILABLE；HTTP 200 不是成功。用户在主账号 Edge 完成 rethos/router → nfnx/router-fc 绑定，随后 API 查询确认成功。已有 FC/ACR 权限足够，不追加管理员权限。
2. 云端模型镜像下载源先后返回 HTTP 500/503，未换模型冒充原快照。改为本机固定版本下载并校验，再分块上传 NAS，每个文件云端复核摘要。
3. 第一版 cb541f6 初始化成功，首个 ready 请求耗时 196.234 秒；第一次固定语料业务请求返回 502。真实复核网关返回 422/trusted_user_id_required：缺少可信服务身份。新增仅由环境配置的 ROUTER_CHECK_USER_ID，经 X-Rethos-User-ID 发往网关；不从用户任务或客户端 Header 取值。真实上游最小请求修复后 200，后续使用新镜像重新验收。
4. 首次失败响应、模型下载失败、后续结果分开保存在忽略 Git 的 output；不把失败轮次写成全部成功。
5. 更新修正版镜像后立即调用曾返回 412/PullImageFailed，提示 accelerated image not ready。等待平台镜像准备完成后重新进行只读就绪检查；此失败没有发送业务复核请求。

## 本地验证

28 项 unittest 通过：既有 19 项路由/历史策略/约束/索引测试、7 项 FC HTTP 契约测试、2 项服务身份测试。服务身份测试使用真实本地 HTTP 服务捕获请求，验证固定身份与恰好一次请求；另以真实生产模型网关验证身份修复。未将本地假向量测试说成真实 GPU 推理。

## 运行与交付边界

单实例并发 1，最小实例数 0。函数支持弹性回收/复用，NAS 持久保存权重和索引；没有承诺固定空闲保留时间、零冷启动、GPU 库存保障或长期负载 SLA。

临时 CPU 传输函数 router-assets-prepare-v3-20261105 及其触发器已删除；此前 v1/v2 也已删除。NAS 资产保留供正式服务使用，没有按整个账号清空业务实例。

Postman 使用本地集合与受 ACL 保护的私密环境，不上传云端，不在集合中写入 Token。公开 HTML、OpenAPI、完整请求示例和私密 Token 预填 HTML 分开交付；私密交付与管理/上游凭据均忽略 Git/Docker。

## 最终云端验收

最终镜像 `router-fc-v2-b454a1b`，digest `sha256:b10e4bdb9f0beb89092c3c9181925cad26618d5cb56ab84a26a291a9ab0d9dd4`，源码 `b454a1b`；ACR 构建 `01A1AC2A-4E1B-5B03-A1BC-E3F8C31A2569` 为 SUCCESS。FC Active / Successful，环境中的部署 revision 与源码一致。

| 验证 | 实际结果 |
| --- | --- |
| 本地 unittest | 28 通过 |
| 真实 FC 完整检查 | 23/23 通过；其中 7 个真实业务请求 |
| Postman 本地 CLI | 15 请求、33 断言通过，退出码 0，27.2 秒，无云上报 |
| 技能 ID | 本轮返回的 5 个不同 ID 全部存在于 734 条注册表，数量上限 1 生效 |
| 约束与降级 | GLM / DeepSeek 约束保留，速度优先返回 codex/deepseek；无关翻译任务返回空技能和默认组合 |
| 错误边界 | 空对象、额外字段、空白、32001 字符、错误类型、重复键为 400；正文超 1 MiB 为 413；错误类型 415；路径/方法 404/405；缺少与错误 Token 为 403 |
| OXR 独立复查 | 就绪 200；真实表格 PNG 同步解析 200/code=0，2.469 秒 |
| 资源收尾 | FC 函数列表只有 rethos-oxr-demo、rethos-router，无本轮临时资产函数残留 |

真实业务样本覆盖固定语料、深度研究、研究报告、科学分析、GLM 约束、DeepSeek 低延迟约束及无关任务。返回的技能包含 tooluniverse-literature-deep-research、scientific-writing、statsmodels、exploratory-data-analysis 和 parallel-web。固定语料及无关任务可返回空技能，属于合法语义。

首次就绪为 196.234 秒；修正版等待镜像加速准备后，就绪请求 7.062 秒，NAS 索引已存在。实例从 `c-6aec94d0-158532d8-1c48b723aca0` 变为 `c-6aec974d-158532d8-962eacf4cbdb`，新实例业务调用成功；证明本轮跨实例读取持久资产可用，不证明所有扩容情形或库存均有保障。主回归业务请求 2.328–13.640 秒，Postman 热业务约 2.8–6.1 秒。没有进行长期或高并发压测。

已检查公开 HTML UTF-8、OpenAPI 四个操作与 15 个集合请求；没有完成浏览器视觉验收。`postman collection lint` 对 v2 JSON 显示 Scanned:0，因此不将其视为完整契约 lint 通过；实际 JSON 解析与集合执行通过。CLI 未登录、未初始化云工作区，status 的技能仓库检查 ECONNRESET；不影响本地集合的真实 HTTP 执行，也没有声称已上传 Postman 云工作区。

完整脱敏回执在 `output/router-live/`，资产与清理回执在 `output/cloud-assets/`。公开调用说明与集合已交付，Token 预填的私密 HTML/环境仅留本机受 ACL 保护路径。
