# PrefixMem

三阶段大模型个性化框架，对应仓库内论文 *Designing Personalized GenAI Services at Scale: A Hierarchical Preference Representation Approach*。

1. `task_LoRA.py`：使用 `user_others.json` 中的 query/answer 训练全局 LoRA。
2. `cluster_profiles.py` + `task_LoRA_group.py`：根据历史画像聚类，冻结并合并全局 LoRA，训练各组 LoRA。
3. `task_LoRA_local_memory.py`：冻结并合并前两阶段，只用目标用户的 `profile` 历史训练用户 soft prefix 和组内共享的 LoRA mediator。目标用户的 query/answer 留给评估。

`eval.py`、`eval_group.py`、`eval_local.py` 分别评估三个阶段。代码源自 OPPU，保留其 LaMP 数据格式及任务模板；TAP-PER 用作本地数据来源。

## 在本机运行

```bash
cd /home/ubuntu/repos/PrefixMem
conda activate perpcs

# 已准备好七个任务的数据和本机兼容依赖，直接执行：
python run_pipeline.py --smoke --task_name movie_tagging
```

默认使用 `/cfs/models/llama/llama3.1-8B` 和一张 GPU。冒烟模式保留原有五组，每组抽取一个共享训练用户和一个目标用户，每个用户最多两条 query、两条历史；全局及各组各训练两步，然后逐阶段推理和评分。真实 8B 模型用于训练，结果只证明流程可运行，不能用于比较模型质量。

输出保存在 `runs/smoke-movie_tagging/`，其中 `logs/` 为六个阶段的日志，`ckpt/` 为全局/组/mediator/用户前缀参数，`output*/` 为预测，`summary.json` 为各阶段 loss、预测数量和指标。重复运行时请用新的 `--run_dir`，以免覆盖已有 checkpoint。

```bash
# 完整数据训练：每卡 batch=1，三个 epoch，默认复用已导入的五组映射。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_pipeline.py \
  --task_name movie_tagging --nproc 8 --run_dir runs/full-movie

# 改用另一个任务或骨干模型。
python run_pipeline.py --task_name scholarly_title --smoke \
  --model_name /cfs/models/llama/llama3.1-8B --run_dir runs/smoke-title

# 有限步数调试（max_step 对每个组单独生效）。
python run_pipeline.py --task_name movie_tagging --max_step 20 --run_dir runs/debug-movie

# 某阶段中断后，可从该阶段续跑；参数需与原运行保持一致。
python run_pipeline.py --task_name movie_tagging --smoke \
  --run_dir runs/smoke-movie_tagging --start_at 21_group_eval
```

全量训练和论文指标复现尚需单独运行；不要将冒烟 checkpoint 当成正式训练结果。默认离线读取本地模型，如需使用 Hugging Face 模型 ID，请设置 `HF_HUB_OFFLINE=0`。

## 数据

`prepare_data.py` 从相邻 TAP-PER 仓库复制七个必要 JSON 文件，保留原划分，不修改 TAP-PER。现有目标文件内容不同时会报错。`data/manifest.json` 记录来源、文件 SHA256、样本数量以及发现的历史/评测重复内容。

```bash
python prepare_data.py
python prepare_data.py --source /path/to/processed-lamp --output ./data
python prepare_data.py --source ./data --validate_only
```

Reviewer history-sparsity cohorts can be prepared with:

```bash
python prepare_history_cohorts.py --source ./data \
  --output ./runs/movie-history-cohorts --task_name movie_tagging
```

This writes independent `short/`, `medium/`, and `long/` run directories. The
short cohort contains the 100 shortest non-focal users, the medium cohort is
the central 100 users after sorting the non-focal pool by history length, and
the long cohort is the published focal split. The selected short and
medium users are removed from the common Stage 1/2 training pool so all three
cohorts use the same shared model data. The large shared training JSON files are
materialized once under `_shared/` and linked into each cohort to avoid
duplicating them. `selection_manifest.json` and each cohort's
`data/history_cohort.json` record the selection and validation statistics.

| 任务 | 共享训练用户 | 目标用户 | 评测 query |
| --- | ---: | ---: | ---: |
| citation | 6,442 | 100 | 123 |
| movie_tagging | 829 | 100 | 3,302 |
| news_categorize | 274 | 49 | 6,033 |
| news_headline | 1,543 | 100 | 6,725 |
| product_rating | 19,899 | 101 | 112 |
| scholarly_title | 14,581 | 101 | 107 |
| tweet_paraphrase | 13,337 | 100 | 109 |

每个任务包含 `user_others.json`、`user_top_100_history.json`、`user_top_100_history_label.json`、`profile_user_others.json`、`profile_user_100.json`、`group_mapping_others.json`、`group_mapping_100.json`。文件名中的 `100` 不是强制用户数。

验证覆盖共享训练用户/目标用户互斥、query ID 唯一、画像/分组完整覆盖，以及评测标签的一一对应。原数据中发现 movie_tagging 5 条、news_categorize 1 条、product_rating 3 条评测输入及答案与同一用户历史完全重复，ID 见 manifest；当前保留源数据，正式实验应另行确定去重协议。检查不代表对所有可能的语义泄漏完成审计。

## 环境与聚类

本机 `perpcs` 的原始 PyTorch 2.0.1 无法执行当前 Llama 的 BF16 掩码；原始 Transformers 4.45.2 也不支持 Qwen3。使用两层可移除的依赖目录，避免修改共享 conda 环境：

- `.deps/runtime`：PyTorch 2.5.1/CUDA 12.4 及匹配依赖；本机链接到 `/cfs/ubuntu/prefixmem-runtime.*`，节省系统盘空间。`run_pipeline.py` 自动加载。
- `.deps/cluster`：Transformers 4.51.3 及 Qwen3 所需依赖，仅在聚类时加载。

重新准备依赖（需要数 GB 空间，可将 `.deps/runtime` 链接到大容量磁盘）：

```bash
python -m pip install --target .deps/runtime -r requirements-runtime.txt
python -m pip install --target .deps/cluster --no-deps -r requirements-cluster.txt
```

新建独立环境时，训练/评估可安装精简后的 `requirements.txt`。如环境已有 torchvision/torchaudio，它们必须与 PyTorch 匹配，见 `requirements-runtime.txt`。

默认复用导入的分组文件（其元数据记录 Qwen3-Embedding-0.6B、K=5）。重新聚类会先复制数据到本次运行目录，再只修改本次运行目录中的映射：

```bash
python run_pipeline.py --task_name movie_tagging --recluster --run_dir runs/recluster-movie

# 单独调用底层脚本时需显式加载依赖。
export PYTHONPATH="$PWD/.deps/runtime${PYTHONPATH:+:$PYTHONPATH}"
PYTHONPATH="$PWD/.deps/cluster:$PYTHONPATH" python cluster_profiles.py --task_name movie_tagging
torchrun --standalone --nproc_per_node=8 task_LoRA.py --task_name movie_tagging
torchrun --standalone --nproc_per_node=8 task_LoRA_group.py --task_name movie_tagging
torchrun --standalone --nproc_per_node=8 task_LoRA_local_memory.py --task_name movie_tagging --group_mode 1
python eval_local.py --task_name movie_tagging --group_mode 1
```

画像生成不在默认运行链中：TAP-PER 已提供画像。聚类按论文使用 masked mean pooling、L2 normalization、K-means；目标用户仅预测分组。重新聚类后必须重训后续组模型和本地记忆。

## 实现说明与验证

保留原代码的 LoRA 设置：全局 rank=8、组 rank=4、mediator rank=4；原 target_modules 中的 `out_proj` 在 Llama 中不匹配，实际命中 q/k/v。论文 EC.2.7 描述全局/组 rank=8、q/v，因此当前可运行实现与论文参数并不完全一致，本次没有擅自改变这些实验设置。prefix length=8，Llama hidden size=4096，每个用户 32,768 个参数。

修复包括：训练步数参数生效、保留答案的截断与 label masking、非重入梯度检查点、阶段三只加载 tokenizer 做预处理、缓存按用户及配置区分并同步 DDP 读写、逐组释放模型、BF16 前缀类型一致、合并 LoRA 后清理 PEFT 0.5 的残留配置、按生成 token 提取预测，以及无需其他仓库/联网的 ROUGE。默认第三阶段读取第二阶段组 LoRA，训练指标额外记录前缀和 mediator 参数是否实际更新。

```bash
PYTHONPATH="$PWD/.deps/runtime" python -m unittest discover -s tests -v
```

回归检查使用本地 Llama tokenizer 和导入的 movie_tagging 数据。已通过单卡 movie_tagging、双卡 scholarly_title 的完整三阶段冒烟验证及8项回归测试，详见 [本机验证记录](VALIDATION.md)；原始训练/推理结果保留在 `runs/` 下各次验证的 `summary.json` 和日志。

## 文件与目录索引

### 核心训练与评估

| 文件/目录 | 作用 |
|---|---|
| `run_pipeline.py` | 串联数据准备、三阶段训练、三阶段评估和结果汇总的入口。 |
| `task_LoRA.py` | Stage 1：训练全局 task LoRA。 |
| `cluster_profiles.py` | 使用用户画像做 K-means 分组，并生成用户到 segment 的映射。 |
| `task_LoRA_group.py` | Stage 2：在全局 LoRA 基础上训练 segment-level LoRA。 |
| `task_LoRA_local_memory.py` | Stage 3：训练用户 soft prefix 和组内 mediator。 |
| `eval.py` / `eval_group.py` / `eval_local.py` | 分别评估 Global、Group、Local 三个阶段。 |
| `gen_profile.py` | 画像生成兼容入口；默认实验直接复用 TAP-PER 提供的画像。 |
| `utils.py` | 任务名称、数据路径、模板和通用辅助函数。 |
| `prepare_data.py` | 从 TAP-PER 复制并验证七个任务的数据文件。 |
| `eval/` | LaMP 任务的指标计算和评估器实现。 |
| `prompt/` | 任务 prompt 和 profile prompt 模板。 |

### Reviewer 实验脚本

| 文件 | 作用 |
|---|---|
| `prepare_history_cohorts.py` | 构造 Short、Medium、Long history cohort。 |
| `prepare_history_budget.py` | 对同一批 Long 用户生成嵌套 25%、50%、75% history 子集。 |
| `run_history_budget.py` | 固定 Stage 1/2，仅重训各 history budget 的 Stage 3。 |
| `summarize_history_budget.py` | 汇总四档 history 结果、用户级 bootstrap CI、斜率和 SVG 图。 |
| `aggregate_main_table_multiseed.py` | 汇总七任务多 seed 主表。 |
| `run_main_table_multiseed.py` | 批量运行主表的多 seed 实验。 |
| `analyze_user_paired_ci.py` | 按用户重采样，计算三阶段配对 bootstrap CI。 |
| `analyze_cohort_stage_ci.py` | 计算 Short/Medium/Long cohort 内的阶段增益 CI。 |
| `analyze_cohort_oppu_ci.py` | 计算 PrefixMem Local 与 OPPU 的同用户配对 CI。 |
| `clean_oppu_predictions.py` | 清洗 OPPU 生成结果后再评估。 |
| `eval_oppu_adapters.py` | 评估 OPPU adapter 及其 RAG/PAG 变体。 |
| `run_k_sensitivity.py` | 运行 K=3/5/7/10 的聚类敏感性实验。 |
| `eval_soft_routing.py` | 评估 K=5 hard/soft routing 性能、激活模块数和延迟。 |
| `summarize_k_sensitivity.py` | 汇总 K sensitivity 和 routing 结果。 |
| `scale_cost_analysis.py` | 计算用户规模增长下的训练、存储和 history exposure 成本。 |
| `scale_k_growth_analysis.py` | 计算固定 K 与随用户数增长的 K 成本敏感性。 |

### Review 文档与结果

| 文件/目录 | 作用 |
|---|---|
| `reviewer_five_point_response.md` | 将五项 reviewer TODO 整理为五个问题和五个回答。 |
| `reviewer_todo_status.md` | 记录五项 TODO 的完成状态和剩余限制。 |
| `history_sparsity_review_response.md` | History sparsity 的 rebuttal 草稿。 |
| `history_sparsity_main_table_cohorts.md` | Short/Medium/Long 及横向 baseline 主表。 |
| `history_budget_controlled_results.md` | 25/50/75/100% 受控 history-budget 结果。 |
| `history_cohort_stage_ci.md` | 自然 cohort 的阶段增益 CI。 |
| `paired_bootstrap_long_3seed.md` | 七任务 Long 三 seed 用户级配对 CI。 |
| `paired_bootstrap_cohort_oppu.md` | Short/Medium Local 与 OPPU 的配对 CI。 |
| `main_table_long_3seed_complete.md` | 七任务 Long 主指标三 seed 均值和标准差。 |
| `k_sensitivity_soft_routing_results.md` / `.json` | K sensitivity、hard/soft routing 和 latency 结果。 |
| `history_scaling_cost_analysis.md` | N=100 到 10,000 的固定 K 成本外推。 |
| `k_growth_cost_sensitivity.md` | K 随用户规模增长时的成本敏感性。 |
| `runs/` | 机器可读的 summary、prediction、CSV、JSON、日志和图；不应提交 checkpoint。 |

### 测试、依赖和论文材料

| 文件/目录 | 作用 |
|---|---|
| `tests/` | history budget、CI、成本敏感性和回归测试。 |
| `requirements.txt` | 精简运行依赖。 |
| `requirements-runtime.txt` / `requirements-cluster.txt` | 训练评估和聚类所需的扩展依赖。 |
| `VALIDATION.md` | 本机冒烟运行和回归验证记录。 |
| `POM-Jun-26-SI-1431_Proof_hi.pdf` | POM 论文 proof，作为论文版本参考。 |
| `Review Report for POM.docx` | 第一轮审稿意见原文。 |
| `To Do List-Heng(1).docx` | 审稿意见对应的实验 TODO。 |
| `tapper.pdf` | TAP-PER 论文参考材料。 |
