# DrugToolAgent LangGraph 试验智能体

用于主环境中测试：由远端 `gpt-5` 模型自主规划并调用各生成模型采样服务（不再做结果文本总结，而是产出调用 JSON 计划并执行）。

## 功能概述
- 调用本地 FastAPI 服务：`DiffSBDD`(端口 3002) 等。
- LangGraph 三节点：`planning` (LLM 生成 JSON 计划) -> `execute` (执行采样) -> `finalize`。
- 使用 OpenAI 兼容接口: `https://kfcv50.link`，模型 `gpt-5` 输出严格 JSON 指令。

## 目录结构
```
agent/
  environment.yaml   # 独立 Conda 环境依赖
  run_agent.py       # 智能体主程序
  README.md          # 说明文档
```

## 环境安装
```bash
conda env create -f agent/environment.yaml
conda activate drugtoolagent
```

或在已有主智能体环境中：
```bash
pip install langchain langgraph httpx pydantic python-dotenv rich typer
```

## 配置密钥 (.env)
推荐使用 dotenv 文件集中管理：
```bash
cp agent/.env.example agent/.env
vim agent/.env   # 编辑填入真实密钥
```
示例内容：
```
AGENT_LLM_API_KEY=sk-your-real-key
LLM_API_BASE=https://kfcv50.link
```
运行脚本会自动加载 `agent/.env`；若不存在则尝试当前工作目录的 `.env`。

## 运行前准备
1. 确保已经启动对应模型服务：
   - DiffSBDD: `bash services/diffsbdd_api/run.sh` (端口 3002)
2. 设置 LLM API Key（你给出的 key），推荐使用环境变量：
```bash
export AGENT_LLM_API_KEY="sk-xxxx...你的key..."
# 可选：修改 base
export LLM_API_BASE="https://kfcv50.link"
```

或直接使用前述 `.env` 文件，无需再 export。

## 使用示例
### 自主规划示例
最简单：
```bash
python agent/run_agent.py --query "给我一些蛋白口袋的起始配体" --show-plan
```

LLM 会返回类似：
```json
{
  "tool": "diffsbdd",
  "args": {
    "pdbfile": ".../2z3h_out.pdb",
    "n_samples": 2,
    "sanitize": true
  }
}
```
随后自动执行对应采样端点并打印返回 JSON。

## JSON Schema（LLM 输出约束）
```json
{
  "tool": "diffsbdd" | "decompdiff" | "midi" | "genmol" | "diffgui",
  "args": { /* 模型参数键值 */ }
}
```
DiffSBDD 可选键：checkpoint, pdbfile, n_samples, batch_size, resi_list, ref_ligand, sanitize, relax, resamplings, jump_length, timesteps

## 扩展规划
后续可在此基础上：
- 增加多模型并行节点（并行采样 + 汇总）。
- 加入打分/过滤节点（如对接 docking 或合成可行性）。
- 引入记忆与向量存储，实现迭代优化循环。

## 故障排查
- 提示端口连接失败：确认服务已启动，或修改 run_agent.py 中 endpoint。
- 规划失败：检查密钥是否正确，或查看输出的原始 JSON 是否格式错误。
- 采样结果为空：确认文件路径、模型权重兼容；可重新运行并让 query 更明确（如指定“使用 DiffSBDD”）。

## 安全与密钥
- 建议永远通过环境变量注入 key，不直接写入代码。
- 若需要在多节点部署，请结合 dotenv 或密钥管理服务。

## 元数据
- 智能体名称：DrugToolAgent Minimal
- 版本：0.1.0
- 框架：LangGraph + LangChain
- 外部 LLM：OpenAI 兼容接口（模型 gpt-5）
