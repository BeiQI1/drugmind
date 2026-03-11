# DrugToolAgent 系统架构文档

本文档基于 `streamlit_app.py` 和 `agent/interactive_workflow.py` 的实际代码逻辑，详细解析 DrugToolAgent 的系统架构、数据流向及交互机制。

## 1. 系统概览 (System Overview)

DrugToolAgent 是一个交互式的 AI 药物发现平台，采用 **LangGraph** 编排多智能体工作流，并通过 **Streamlit** 提供可视化交互界面。系统通过共享状态 (`AgentState`) 和动态路由 (`Router`) 实现灵活的任务规划与执行。

---

## 2. 核心交互层 (Interaction Layer)

### 2.1 用户界面 (`streamlit_app.py`)
这是系统的入口，负责用户交互、文件管理和实时反馈。

*   **会话状态管理 (`st.session_state`)**:
    *   `messages`: 存储用户与 AI 的完整对话历史。
    *   `uploaded_files`: 管理用户上传的 PDB 文件或分子 CSV 文件，路径存储于 `data/uploads`。
    *   `run_id`: 为每次任务生成唯一标识符，用于追踪日志和输出文件。
    *   `intent_turn_id`: 用于处理多轮对话中的意图确认。

*   **实时日志系统 (`StreamLogger`)**:
    *   **机制**: 劫持标准输出 (`stdout`)，实时捕获后端 Agent 的打印信息。
    *   **解析**: 通过正则匹配 `[AgentName]` 标签，将日志实时渲染到 Streamlit 的状态容器 (`st.status`) 中，实现“所见即所得”的执行过程展示。

### 2.2 工作流引擎 (`agent/interactive_workflow.py`)
这是系统的核心调度器，基于 LangGraph 定义了图结构的工作流。

*   **状态定义 (`AgentState`)**:
    继承自 `TypedDict`，作为所有 Agent 共享的黑板：
    *   `plan`: 执行计划列表 (如 `["TargetAgent", "GeneratorAgent"]`)。
    *   `plan_step`: 当前执行步骤索引。
    *   `results`: 存储各阶段产物 (如生成的 SMILES、对接打分)。
    *   `task_params`: 存储任务参数及累计的 `agent_logs`。

*   **节点封装 (Node Wrappers)**:
    代码中定义了如 `intent_node_wrapper`, `target_node_wrapper` 等包装函数。这些 Wrapper 不仅调用 Agent 核心逻辑，还负责：
    1.  **日志格式化**: 将执行结果格式化为 Markdown 字符串并写入 `task_params["agent_logs"]`。
    2.  **状态更新**: 递增 `plan_step` 或 `loop_count`。
    3.  **CLI 反馈**: 打印带有 `[AgentName]` 前缀的日志供前端捕获。

---

## 3. 动态路由机制 (Dynamic Routing)

系统采用“意图识别 -> 规划生成 -> 动态分发”的三段式路由逻辑。

1.  **意图路由 (`route_after_intent`)**:
    *   `START` -> `IntentAgent`
    *   根据识别出的 `intent` 判断：
        *   若为明确任务 (如 `generation`, `evaluation`) -> 进入 `CoordinatorAgent`。
        *   若需要澄清 (`clarification_needed`) 或出错 -> 直接结束 (`END`) 并返回询问用户。

2.  **规划执行路由 (`plan_router`)**:
    *   `CoordinatorAgent` 生成 `plan` (Agent 名称列表) 后，控制权交给 `plan_router`。
    *   **逻辑**: 读取 `state["plan"]` 和 `state["plan_step"]`。
    *   **分发**: 
        *   若 `step < len(plan)` -> 路由到对应的 Agent (如 `TargetAgent`, `GeneratorAgent`)。
        *   Agent 执行完毕后 -> 再次回到 `plan_router`。
        *   若 `step >= len(plan)` -> 路由到 `END`，任务完成。

---

## 4. 智能体节点 (Agent Nodes)

每个节点在 `interactive_workflow.py` 中被注册为图的一个顶点。

| 节点名称 | 对应 Agent 类 | 核心职责 |
| :--- | :--- | :--- |
| **IntentAgent** | `agent/IntentAgent.py` | 解析自然语言，提取任务参数 (PDB ID, SMILES)，初始化 `task_params`。 |
| **CoordinatorAgent** | `agent/CoordinatorAgent.py` | **大脑**。根据 Intent 和当前资源，生成有序的 Agent 执行列表 (`plan`)。 |
| **TargetAgent** | `agent/TargetAgent.py` | 处理靶点数据。下载 PDB，识别结合口袋，输出处理后的靶点路径。 |
| **GeneratorAgent** | `agent/GeneratorAgent.py` | 调用生成模型 (DiffSBDD/GenMol)。支持循环生成 (`loop_count`)。 |
| **EvaluatorAgent** | `agent/EvaluatorAgent.py` | 评估分子性质 (Docking, QED, SA)。统计合格分子数量。 |
| **SynthesisAgent** | `agent/SynthesisAgent.py` | 逆合成分析。规划合成路径，评估合成难度。 |
| **ReportAgent** | `agent/ReportAgent.py` | **收尾**。汇总 `task_params["agent_logs"]` 和 `results`，生成 PDF 报告。 |

### 辅助组件 (Utility Components)

除了上述核心节点外，系统还包含辅助智能体以支持特定功能：

*   **RAGAgent** (`agent/RAGAgent.py`):
    *   **角色**: 知识检索专家。
    *   **功能**: 维护基于 FAISS 的向量数据库，索引 `services/API_REFERENCE.md` 和 `drugtoolkg/agent_kg.json` 等文档。
    *   **调用**: 目前主要被 `CoordinatorAgent` 内部实例化使用，用于在规划阶段检索工具用法或 API 细节，增强 LLM 的上下文理解能力。它不直接参与 LangGraph 的路由跳转，而是作为服务被其他 Agent 调用。

---

## 5. 数据流转示例 (Data Flow Example)

假设用户输入：**"基于 1XYZ.pdb 生成分子并评估"**

1.  **Streamlit**: 初始化 State，捕获输入，调用 `app.invoke`。
2.  **IntentAgent**: 识别 Intent=`generation`，提取 PDB ID=`1XYZ`。
3.  **CoordinatorAgent**: 生成 Plan=`["TargetAgent", "GeneratorAgent", "EvaluatorAgent", "ReportAgent"]`。
4.  **TargetAgent**: 下载 `1XYZ.pdb` -> 存入 `results["target_preparation"]` -> `plan_step=1`。
5.  **GeneratorAgent**: 读取 PDB -> 生成分子 -> 存入 `results["generation"]` -> `plan_step=2`。
6.  **EvaluatorAgent**: 读取分子 -> 运行 Docking -> 存入 `results["evaluation"]` -> `plan_step=3`。
7.  **ReportAgent**: 读取所有结果 -> 生成 PDF -> `plan_step=4`。
8.  **Router**: 检测 `step=4` (计划结束) -> `END`。
9.  **Streamlit**: 显示最终回复和下载链接。

---

## 6. 文件结构映射 (File Mapping)

*   `streamlit_app.py`: **前端入口** (UI, Session, Logger)。
*   `agent/interactive_workflow.py`: **后端引擎** (LangGraph 定义, Node Wrappers, Routers)。
*   `agent/state.py`: **数据协议** (AgentState 类型定义)。
*   `agent/RAGAgent.py`: **知识引擎** (向量检索服务)。
*   `agent/*.py`: **业务逻辑** (各功能 Agent 的具体实现)。
*   `data/uploads/`: **临时存储** (用户上传的文件)。
*   `agent/outputs/`: **结果输出** (生成的报告、分子文件)。

