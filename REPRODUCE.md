# LoCoMo 实验复现说明

给合作者复现这个仓库里的 LoCoMo 实验时，按下面做即可。

## 1）先准备什么

1. 仓库代码

- `a-mem-memgit`

2. 数据集

- `data/locomo10.json`

3. 安装依赖

在仓库目录下执行：

```bash
pip install -r requirements.txt
```

## 2）怎么配参数

主入口脚本：

- `scripts/run_locomo_baseline_patch_3models.sh`

主要改这个文件顶部的配置：

```bash
CONFIG_API_KEY=""
CONFIG_API_BASE="https://app.ppapi.ai/v1/chat/completions"
CONFIG_RUN_TARGET="robust"
CONFIG_MODELS=(
  "gpt-5.4-mini"
)
CONFIG_RATIO="1.0"
CONFIG_START_SAMPLE="0"
CONFIG_BATCH="1"
CONFIG_RAW_LLM_LOG=""
```

说明：

- `CONFIG_RUN_TARGET`
  - `robust`：baseline
  - `patch`：patch 版本
  - `both`：两个都跑
- `CONFIG_MODELS`
  - 想跑哪些模型就写哪些
- `CONFIG_RATIO="0.1"`
  - 表示只跑前 10%，适合 smoke test
- `CONFIG_START_SAMPLE="0"`
  - 默认从第一个 sample 开始
- `CONFIG_BATCH="1"`
  - 最稳妥，建议先用这个
- `CONFIG_RAW_LLM_LOG`
  - 只有在排查接口问题时才需要开

更推荐的做法是不要把 key 写进脚本，而是在运行前设置：

```bash
export OPENAI_API_KEY="your_key"
```

## 3）怎么跑实验

### 跑 baseline

把脚本里设成：

```bash
CONFIG_RUN_TARGET="robust"
```

然后执行：

```bash
bash scripts/run_locomo_baseline_patch_3models.sh
```

### 跑 patch

把脚本里设成：

```bash
CONFIG_RUN_TARGET="patch"
```

然后执行：

```bash
bash scripts/run_locomo_baseline_patch_3models.sh
```

### baseline 和 patch 都跑

把脚本里设成：

```bash
CONFIG_RUN_TARGET="both"
```

然后执行：

```bash
bash scripts/run_locomo_baseline_patch_3models.sh
```

注意：当前脚本是串行执行，不会自动并行开多个 tmux session。

## 4）怎么看结果

结果文件位置：

- baseline 结果：`robust_results/`
- patch 结果：`patch_results/`

日志位置：

- `logs/`

常用查看方式：

```bash
ls -lt robust_results
ls -lt patch_results
ls -lt logs | head
```

如果想实时看日志：

```bash
tail -f logs/<log_file>.log
```

## 5）推荐的最小测试方式

如果只是先确认能不能跑通，建议先把脚本设成：

```bash
CONFIG_RUN_TARGET="patch"
CONFIG_MODELS=(
  "gpt-5.4-mini"
)
CONFIG_RATIO="0.1"
CONFIG_START_SAMPLE="0"
CONFIG_BATCH="1"
```

然后运行：

```bash
bash scripts/run_locomo_baseline_patch_3models.sh
```

## 6）提醒

对外分享前：

- 不要提交真实 API key
- `CONFIG_API_KEY` 保持为空
- 不要提交 `.env`
- 不要提交 `logs/`、`patch_results/`、`robust_results/`、`cached_memories_*/`
