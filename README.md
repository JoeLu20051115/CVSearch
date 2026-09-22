# MUSE

**Multi-Granularity Visual Search with Verification Feedback**，用于高分辨率图像问答的冻结模型推理方法。

MUSE 先用生成器检查全图答案的置信度与选项间隔；需要继续观察时，通过 SAM 3 和 SGAP 获取局部视图。生成器与独立验证器读取全图及全部已取得的局部视图，验证器逐选项给出首 token 支持分数和同次生成的语义反馈，驱动后续观察与停止。搜索耗尽时返回预先保存的全图答案，并标记为未经验证的回退。

本仓库提供核心搜索、模型适配与单图命令。[方法附录](docs/appendix.md) 依据用户提供的 38 页 `ICLR27_MUSE.pdf`，列明公式、固定提示、论文参数和实现接口。论文未公布部分运行参数；执行前须显式提供，不能仅凭该 PDF 声称数值完全复现或重现其报告的实验性能。

## 安装

使用 Python 3.11 或更新版本，以及与 PyTorch 2.7 兼容的 CUDA 环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SAM 3 与 LLaVA 依赖由 `requirements.txt` 固定到上游提交。模型权重放在仓库外。论文使用 InternVL2.5-8B、LLaVA-OV-7B 或 Qwen2.5-VL-7B 作为生成器，默认独立验证器为 **Qwen3-VL-4B-Instruct**；另需 SAM 3 与 CLIP 权重。

## 单图推理

```bash
python -m muse \
  --image /path/to/image.jpg \
  --question "What color is the man's hat?" \
  --options red blue green black \
  --generator /path/to/Qwen2.5-VL-7B-Instruct \
  --verifier /path/to/Qwen3-VL-4B-Instruct \
  --sam /path/to/sam3.pt \
  --clip /path/to/clip-vit-large-patch14 \
  --runtime-config /path/to/runtime.json \
  --device cuda:0
```

存在性问题使用 `--options Yes No`。`--verifier-device cuda:1` 可将验证器放在另一设备。输出 JSON 区分全图门接受、局部验证接受及未经验证的回退；回退答案不等于通过了证据检查。

运行 JSON 的所有顶层字段均必填，名称与 [RuntimeConfig](muse/config.py) 一致：

| 字段 | 含义 |
| --- | --- |
| `global_confidence`, `global_margin` | 全图生成器门的两阈值，均在 `(0,1]`。 |
| `planning_tokens`, `navigation_tokens`, `verifier_tokens` | 对应生成过程的输出 token 上限。 |
| `generator_context_tokens`, `verifier_context_tokens` | 两模型允许的完整输入与输出上下文容量。 |
| `edge_size` | Sobel 排序图尺寸 `[height, width]`，两项至少为 3。 |
| `edge_interpolation` | `bilinear`、`bicubic` 或 `lanczos`。 |
| `sgap` | 完整 SGAP 参数对象；字段见[附录](docs/appendix.md#g-论文参数与运行配置)。 |

这些字段的数值未在论文中完整披露，须使用实际运行记录或明确的工程配置。论文表 S1 的固定参数直接定义于代码：局部观察上限 8，支持门 0.65，归一化间隔门 0.15，双进展阈值均为 0.01，停滞耐心 2，排序权重 0.70/0.30、特征份额 0.50。命令行不提供调参模式。

## 代码

| 位置 | 内容 |
| --- | --- |
| `muse/search.py` | 全图门、累计证据、局部接受、动作与恢复控制。 |
| `muse/types.py` | 视图、候选、定位与模型输出记录。 |
| `muse/config.py` | 论文固定参数与显式运行配置。 |
| `muse/prompts.py` | 问题规划、回答、导航和逐选项验证提示。 |
| `muse/frontend.py` | SAM、SGAP、CLIP 与候选排序。 |
| `muse/models.py` | 多图输入、首 token logits 与模型容量接口。 |
| `muse/__main__.py` | 单图推理命令。 |
| `docs/appendix.md` | 核心方法说明。 |
| `tests/` | 使用合成输入与模型替身的功能检查。 |

运行功能检查：

```bash
pip install -e '.[test]'
python -m pytest -q
```

## 致谢

区域构建和模型适配使用或参考 [CVSearch](https://github.com/liliupeng28/ICML26-CVSearch)、[ZoomEye](https://github.com/om-ai-lab/ZoomEye)、[SAM 3](https://github.com/facebookresearch/sam3) 和 [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)。
