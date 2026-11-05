# 独立 Router GPU 函数

源码入口 `gateway.py`，镜像入口 `docker/Dockerfile.fc`。不启动 OXR、vLLM 服务，也不改变原命令行 Router。默认容器启动命令由镜像 ENTRYPOINT 决定，监听 9000；不沿用演示镜像的 `/code/run.sh`。

## 已部署资源

- 地域 cn-hangzhou；函数 rethos-router；业务 POST /route。
- ACR 实例 cri-fmgbldj0anwo4c5g，仓库 rethos/router。
- 部署镜像 router-fc-v2-b454a1b，源码 b454a1b，linux/amd64。
- 镜像 digest sha256:b10e4bdb9f0beb89092c3c9181925cad26618d5cb56ab84a26a291a9ab0d9dd4。
- 构建分支 deploy/aliyun-fc；上下文 `/`；Dockerfile `docker/Dockerfile.fc`；关闭自动构建，开启海外源加速，保留缓存。
- 使用已固定 digest 的 ACR vllm-base 作为 CUDA/Torch 运行依赖底座，Router 使用独立 Python venv。基础镜像中的其他工具不会自动成为业务能力。
- NAS `0013700ik5xmikpq4d3` 的 `/router` 挂到 `/mnt/router-assets`。VPC 与 ACR 内网入口相同。不得把 OXR NAS 根目录改成 Router 目录。
- GPU Ampere 24 GB / CPU 8 / RAM 32 GB；并发 1；最小实例 0；执行超时 240；初始化 POST /initialize 660 秒。

## NAS 必需布局

```text
/mnt/router-assets/
  models/ThakiCloud--SKILLRET-Embedding-0.6B/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    ...固定模型快照其余文件
  data/research-skill-pool-v2-20261104/
    MANIFEST.json
    skills.jsonl
    ...完整技能包
  indexes/request-router/skillret.npz    # 首次初始化生成，以后校验后复用
```

技能包 MANIFEST SHA256 `586759346a624fc13f905678359d880f5a26a1ac23bfd34e6198698b95d57031`；主权重 SHA256 在 `config/request_router.fc.json` 固定。权重与技能是受哈希校验的部署资产，忽略 Git 与 Docker 构建上下文。不要用其他 embedding 模型冒充同一快照。

## 私密配置

管理 AK 留在本机受保护部署环境，不写入镜像。FC 环境变量设置：

```ini
ROUTER_CHECK_BASE_URL=<复核模型兼容接口 Base URL，以 /v1 结尾>
ROUTER_CHECK_PROVIDER=deepseek
ROUTER_CHECK_MODEL=<复核模型 ID>
ROUTER_CHECK_API_KEY=<仅复核调用所需业务 Token>
ROUTER_CHECK_USER_ID=router-fc-service
ROUTER_DEPLOYMENT_REVISION=b454a1b
```

独立 FC HTTP 触发器启用 Bearer 鉴权，以 `ROUTER_FC_TOKEN` 作为客户端 Token，不能拿复核模型业务 Token 或阿里云 AK 充当客户端 Token。

`ROUTER_CHECK_USER_ID` 是该独立服务访问网关的固定服务身份，不冒充终端用户；只从受信部署环境设置，不接受任务文本或客户端 Header 覆盖。直连不要求此身份的供应商时可以省略。

## 更新与回退

原始 origin 是组织仓库，personal 是用户副本；从 origin 获取更新。本轮授权同时推送两个仓库。构建新版本使用新 tag，等待 ACR 构建成功和 tag digest 确认后更新 FC；不要覆盖已被 FC 引用的 tag。保留 NAS 权重/技能，配置或快照改变时索引自动重建。

回退时恢复此前的 Router 镜像 tag 和对应配置，保留凭据，不回退或删除 OXR 服务。没有更早 Router 正式版本时可停用 Router 触发器而不删除 NAS。测试临时函数不应长期保留。

## 本地验证与文档生成

```bash
python -m unittest discover -s tests -v
python deploy/fc/build_collection.py
python deploy/fc/build_docs.py
```

预填私密文档需设置 `ROUTER_FC_TOKEN` 并执行 `build_docs.py --private`，生成文件必须维持 ACL 保护与 Git/Docker 忽略。不要打印其全文。

公开调用说明在 `docs/FC接口文档.md`，实际验证与边界在 `docs/FC部署验收.md`。
