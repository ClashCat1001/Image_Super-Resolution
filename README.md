# Real-ESRGAN 本地批量超分网页

这个项目提供一个本地网页前端和 Flask 服务端，用开源 `Real-ESRGAN` 对图片做批量超分，并把结果保存到当前目录下的 `Images_Output` 文件夹。

## 功能

- 批量上传图片
- 输入 HTTP / HTTPS 图片下载 URL，每行一个链接，自动下载后批量超分
- 直接扫描根目录下的 `Images_Input/`，递归识别子文件夹中的图片
- 扫描时自动跳过 `Images_Input/Archive/`
- 目录模式处理成功后，自动把原图移动到 `Images_Input/Archive/` 下的同名子目录
- 选择输出倍率：`2x / 3x / 4x`
- 选择降噪强度：`d0x - d10x`
- 输出文件名格式：`原文件名-3x-d3x.png`
- 处理完成后自动保存到 `Images_Output/` 的对应同名子目录
- 首次运行自动下载官方 `realesr-general-x4v3` / `realesr-general-wdn-x4v3` 权重到 `weights`
- 自动模式依次选择 CUDA、Metal(MPS)、CPU，也支持在网页中手动选择
- Metal 推理异常时自动回退 CPU
- 清晰原图低处理模式：限制降噪并混合 Lanczos 放大结果，减少涂抹感

## 推荐环境

- macOS / Linux / Windows
- Python `3.10` 或 `3.11`
- 首次启动需要联网安装依赖，并在第一次推理时下载模型权重

## 启动方式

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

打开浏览器访问：

```text
http://127.0.0.1:8000
```

## 目录说明

- `Images_Input/`：目录模式的输入目录
- `Images_Input/Archive/`：目录模式成功处理后的原图归档目录
- `app.py`：Flask 服务入口
- `app/realesrgan_service.py`：Real-ESRGAN 推理封装
- `app/image_download.py`：图片 URL 校验、限量下载和临时文件处理
- `templates/index.html`：网页结构
- `static/`：前端样式和交互脚本
- `weights/`：模型权重
- `uploads/`：临时上传目录
- `Images_Output/`：超分结果输出目录

## 使用下载 URL

在网页选择“下载 URL”，粘贴图片文件直链。多个 URL 直接换行分隔，中间不添加逗号、分号或其他符号：

```text
https://example.com/image-1.jpg
https://example.com/image-2.png
```

选择倍率、降噪强度和处理模式后点击“开始超分”，服务会先下载图片，再自动处理并保存到 `Images_Output/`。空行会被忽略；单个链接失败不影响其他图片，失败详情显示在输出结果中。链接应直接返回图片文件，网页链接或需要登录的链接可能无法下载。

每批最多 100 个链接，单张图片最多 50 MB，整批下载图片最多 500 MB。支持 HTTP / HTTPS 跳转及不带图片扩展名的下载链接。下载原图仅临时保存，处理完成或失败后自动清理；输出优先使用下载响应的文件名，否则使用链接中的文件名。

下载由本机 Python 服务使用本机网络完成，并沿用系统代理或 `HTTP_PROXY` / `HTTPS_PROXY` 等环境设置。Pixiv 的 `pximg.net` 图片直链会自动附带 Pixiv 来源信息，以支持该站的防盗链校验；仍需登录或额外权限的图片可能无法下载。

接口 `POST /api/upscale` 使用 `source_mode=url`，将换行分隔的链接放入表单字段 `urls`；倍率、降噪、设备和处理模式沿用现有字段。

## 说明

- 当前实现默认使用 `realesr-general-x4v3`，通过 `outscale` 支持 2x、3x、4x 输出。
- 如果遇到大图显存不足，可以在 `RealESRGANBatchService` 初始化时把 `tile` 改成更小的值，例如 `256`。
- 目录模式会保留 `Images_Input/` 内的子目录结构，并在 `Images_Output/` 和 `Images_Input/Archive/` 中同步复现。
- 当前降噪命名中的 `d3x` 表示降噪等级 `3`，内部会映射成 Real-ESRGAN 的 `0.3` 插值权重。
- 服务会根据图片尺寸自动做外层分块推理，避免大图整图推理导致进程卡死或被系统杀掉。
- CPU 模式会显式启用多线程，默认使用当前机器的全部逻辑核心；可用 `REAL_ESRGAN_CPU_THREADS=8 python app.py` 调整线程数。
- 清晰原图可选择“清晰原图 · 低处理”：实际降噪等级为 `min(所选等级, 2)`，输出为 75% Lanczos 放大图与 25% 模型超分图的混合。保留更多原始纹理，也会保留更多原图噪点；它仍需模型推理，不代表更快的模式。
- 低处理模式保留透明通道的 Lanczos 放大结果，文件名增加 `-low`，例如 `原文件名-3x-d2x-low.png`；标准模式默认行为和命名保持原样。
- 接口 `POST /api/upscale` 新增可选字段 `processing_mode=standard|low`（默认 `standard`）。响应的 `denoise_level` 为实际等级，`requested_denoise_level` 为所选等级，结果包含 `processing_mode` 和 `model_name`。
- 使用有界读图预取与后台写图，让 I/O 与推理重叠；OpenCV 使用单线程，避免与 Torch 的多核推理争抢线程。模型缓存最多保留 3 个配置，推理后释放缓存推理器持有的图像张量，CUDA 缓存清理改为每批一次。
- 同一服务进程的处理请求排队执行，避免共享推理器的输入/输出状态串扰、输出路径冲突及重复归档。请使用单个服务进程；线程锁不跨进程共享。
- 在 Apple Silicon 上，若手动启用 Metal 且输出异常黑图，服务会自动切到 CPU，避免继续写出错误结果。
