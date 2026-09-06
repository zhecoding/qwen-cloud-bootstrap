# 云端 llama.cpp 启动工具

这个项目用于在已经启动的 Vast.ai 或 RunPod RTX 5090 实例中，通过一条终端命令准备 llama.cpp 推理环境、下载所需文件并启动服务。

它不会制作或绑定自定义 Docker 镜像，也不会使用平台的自动 Provisioning 功能。流程是：你先租用 GPU、选择推荐镜像并启动实例，然后进入终端运行本项目。

## 默认配置

| 项目 | 默认值 |
|---|---|
| GPU | 单张 RTX 5090 32 GB |
| 模型量化 | `Q5_K_P`，约 20.22 GB |
| 加速 | FastMTP，约 0.90 GB |
| 视觉 projector | 开启，约 0.93 GB |
| 上下文 | 32,768 tokens |
| 并发 | 1 |
| GPU offload | 全部层 |
| 服务 | `llama-server` |
| 服务端口 | `8000` |
| 接口 | 内置 Web UI、OpenAI-compatible API |
| 持久存储 | 不使用 |

脚本会固定模型仓库 revision、`llama.cpp` commit、FastMTP 补丁以及每个下载文件的大小和 SHA-256。它还会自动生成 API key、等待健康检查，并发送一条真实推理请求确认服务可用。

## 一、Vast.ai 完整教程

### 1. 选择镜像

创建或编辑 Vast.ai 模板，选择下面这个明确版本：

```text
vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310
```

不要选择名称中带 `runtime` 的 CUDA 镜像，因为现场编译 `llama.cpp` 需要 `nvcc`。也不需要选择 PyTorch 或 ComfyUI 镜像。

Vast.ai 的官方 **Llama.cpp** 应用模板也能启动 GGUF 模型，但不推荐用于本项目。该模板自带并自动管理一套标准 `llama-server`，而本项目需要为 FastMTP 应用专用补丁并编译固定版本；它默认还会把外部端口 `8000` 映射到内部端口 `18000`。使用它仍需停用内置服务、重新编译并调整端口，因此不会比上述基础 CUDA 镜像更省事。

### 2. 模板和实例设置

建议设置如下：

| 设置 | 推荐值 |
|---|---|
| GPU | `1x RTX 5090` |
| Disk Space | `60 GB` |
| 系统内存 | 至少 `32 GB`，推荐 `48 GB` 以上 |
| CPU | 至少 `8` 线程 |
| 下载带宽 | 建议 `500 Mbps` 以上 |
| 磁盘速度 | 建议 `500 MB/s` 以上 |
| Reliability | 建议 `98%` 以上 |
| 启动模式 | SSH 或 Jupyter |
| 端口 | 暴露 `8000` |
| Persistent Volume | 不配置 |
| `PROVISIONING_SCRIPT` | 留空 |

端口使用 `8000`，避免与基础镜像中的 Jupyter `8080` 冲突。

[此处](https://cloud.vast.ai/create/?gpuModelNames=rtx5090&instanceDiskSizeMin=49.99999999999996&machineMegabitDownloadMin=1000&machineMegabitUploadMin=1000&machinePortsOpenMin=9&offerGpuNumMax=1&offerGpuNumMin=1&priceInstanceTerabyteDownloadMax=5&priceInstanceTerabyteUploadMax=5)为已包含筛选条件的搜索链接。

### 3. 启动实例并打开终端

等待实例状态变成 Running。然后任选一种方式进入终端：

- 点击 SSH 按钮，使用 Vast.ai 显示的 SSH 命令连接。
- 点击 Jupyter，在 JupyterLab 中打开 Terminal。

首先确认 GPU 和 CUDA：

```bash
nvidia-smi
nvcc --version
```

应当看到 RTX 5090，并且 CUDA 为 12.8 或更高。

### 4. 下载并启动

```bash
git clone https://github.com/zhecoding/qwen-cloud-bootstrap.git /workspace/qwen-cloud
cd /workspace/qwen-cloud
bash bootstrap.sh
```

如果以后把仓库改为私有仓库，需要自行配置访问凭据。

脚本会依次检查环境、编译 `llama.cpp`、下载约 22 GB 模型、校验文件、启动服务并执行测试。根据主机 CPU 和网络速度，第一次运行通常需要数分钟到二十分钟左右。

完成后终端会显示：

- Web UI 地址
- OpenAI API base URL
- 模型名称
- API key
- 日志路径
- PID 文件路径
- 停止命令

### 5. 访问服务

Vast.ai 会给内部端口 `8000` 分配外部端口。在实例卡片的端口信息中找到对应映射，然后访问：

```text
http://VAST_PUBLIC_IP:MAPPED_PORT
```

更安全的方式是使用 SSH 隧道：

```bash
ssh -p SSH_PORT root@VAST_PUBLIC_IP -L 8000:127.0.0.1:8000
```

保持该 SSH 会话打开，然后在本地浏览器访问：

```text
http://127.0.0.1:8000
```

Web UI 在首次调用模型时需要使用安装结束时显示的 API key。

## 二、RunPod 完整教程

### 1. 创建模板

在 RunPod 创建自定义 Pod 模板，镜像填写：

```text
runpod/base:1.0.2-cuda1281-ubuntu2204
```

推荐设置：

| 设置 | 推荐值 |
|---|---|
| GPU | `1x RTX 5090` |
| Container Disk | `60 GB` |
| Volume Disk | `0 GB` 或不配置 |
| Network Volume | 不配置 |
| Expose HTTP Ports | `8000` |
| Docker Start Command | 留空 |
| Docker Entrypoint | 留空 |

该镜像基于 CUDA 12.8.1 development 镜像，包含编译 `llama.cpp` 所需的 CUDA toolkit。这个项目不依赖 PyTorch，因此无需选择体积更大的 PyTorch 模板。

### 2. 启动并进入终端

使用模板创建 RTX 5090 Pod。Pod 运行后，通过 Web Terminal、Jupyter Terminal 或 SSH 进入终端，然后确认：

```bash
nvidia-smi
nvcc --version
```

### 3. 下载并启动

```bash
git clone https://github.com/zhecoding/qwen-cloud-bootstrap.git /workspace/qwen-cloud
cd /workspace/qwen-cloud
bash bootstrap.sh
```

完成后可以通过 RunPod HTTP proxy 访问：

```text
https://POD_ID-8000.proxy.runpod.net
```

`POD_ID` 是 RunPod Pod 的 ID。脚本在识别到 RunPod 环境后也会直接显示这个地址。

## 三、调用 OpenAI-compatible API

查看当前 API key：

```bash
cat /workspace/qwen-runtime/secrets/api-key.txt
```

在云端实例内部测试：

```bash
API_KEY="$(cat /workspace/qwen-runtime/secrets/api-key.txt)"
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-aggressive-q5",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "max_tokens": 512
  }'
```

在本地软件中配置时：

- Base URL：平台公开地址后加 `/v1`
- API key：安装结束时显示的 key
- Model：`qwen3.8-27b-aggressive-q5`

## 四、日志、停止和重新启动

实时查看日志：

```bash
tail -f /workspace/qwen-runtime/logs/llama-server.log
```

查看进程：

```bash
ps -fp "$(cat /workspace/qwen-runtime/run/llama-server.pid)"
```

停止服务：

```bash
cd /workspace/qwen-cloud
bash bootstrap.sh --stop
```

重新启动：

```bash
cd /workspace/qwen-cloud
bash bootstrap.sh
```

同一个实例内重新运行时，已经校验正确的模型和已经完成的编译会被复用。实例销毁后，因为没有持久存储，所有内容都会消失；下次创建新实例时重新运行安装命令即可。

## 五、可选设置和故障排查

### Hugging Face 限流

公开模型通常不要求登录。如果匿名下载遇到限流，可以在运行前设置 Hugging Face token：

```bash
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
bash bootstrap.sh
```

### 自己指定 API key

```bash
export QWEN_API_KEY="YOUR_LONG_RANDOM_API_KEY"
bash bootstrap.sh
```

不要把真实 token 或 API key 写入公开文件。

### FastMTP 无法启动

默认使用 FastMTP。如果某台主机出现相关兼容性问题，改用模型自带的 embedded MTP：

```bash
bash bootstrap.sh --stop
QWEN_MTP_MODE=embedded bash bootstrap.sh
```

### CUDA 或 Flash Attention 错误

RTX 5090 的驱动、CUDA kernel 和主机环境存在差异。如果日志中出现 Flash Attention CUDA 错误，可以关闭它进行排查：

```bash
bash bootstrap.sh --stop
QWEN_FLASH_ATTN=off bash bootstrap.sh
```

### 显存不足

先把上下文从 32K 降到 16K：

```bash
bash bootstrap.sh --stop
QWEN_CTX_SIZE=16384 bash bootstrap.sh
```

如果仍然不足，再临时关闭视觉 projector：

```bash
bash bootstrap.sh --stop
QWEN_CTX_SIZE=16384 QWEN_ENABLE_VISION=0 bash bootstrap.sh
```

### 修改运行目录或端口

```bash
QWEN_DATA_DIR=/workspace/my-qwen-runtime QWEN_PORT=9000 bash bootstrap.sh
```

如果修改端口，还需要在 Vast.ai 或 RunPod 中暴露对应端口。

### 安装失败后重试

下载支持断点续传。大多数网络中断只需再次执行：

```bash
bash bootstrap.sh
```

如果服务启动失败，先查看：

```bash
tail -n 100 /workspace/qwen-runtime/logs/llama-server.log
```

## 六、重要说明

- 该模型是第三方 Aggressive/uncensored 变体，不代表它比官方模型更准确或更可靠。
- 默认配置针对单张 RTX 5090 32 GB；其他 GPU 必须显式设置 `QWEN_ALLOW_OTHER_GPU=1`，并自行确认显存是否足够。
- 不要在公网无保护地分享 API key。
- Vast.ai 直接映射端口通常是 HTTP；需要更强安全性时优先使用 SSH 隧道或平台提供的 HTTPS tunnel。
- 本项目不会自动停止或销毁计费中的实例。使用完成后请回到 Vast.ai 或 RunPod 控制台停止或销毁实例。

## 七、参考资料

- [模型仓库与推荐参数](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF)
- [llama.cpp server 文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Vast.ai base-image 文档](https://github.com/vast-ai/base-image/blob/main/README.md)
- [RunPod 官方镜像构建配置](https://github.com/runpod/containers/blob/main/official-templates/base/docker-bake.hcl)
- [RunPod HTTP 端口文档](https://docs.runpod.io/pods/configuration/expose-ports)
