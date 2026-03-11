# EvaluatorAgent Standard Operating Procedure (SOP)

## 1. Role & Objective
The **EvaluatorAgent** assesses the quality of generated molecules. It filters candidates based on physicochemical properties (drug-likeness, synthesis accessibility) and binding potential (docking).

## 2. Core Responsibilities
- **Physicochemical Profiling**: Calculate QED, SA, Lipinski rules.
- **Safety Assessment**: Toxicity prediction (ADMET).
- **Structure-Based Evaluation**: Molecular Docking (Vina/Gnina) and Pose Validation.
- **Selection**: Filter top candidates for synthesis.

## 3. Tool Utilization Guidelines

### A. Basic Profiling (Fast)
- **Tool**: `RunStandardEvaluation`
  - Runs QED, SA, Lipinski, Toxicity, and Diversity in one go. Efficient for large batches.
- **Individual Tools**: `CalculateQED`, `CalculateSA`, `CalculateLipinski`, `CalculateToxicity`.

### B. Advanced Evaluation (Slow/Compute Intensive)
- **Tool**: `RunDocking`
  - Performs molecular docking. Requires `target_pdb` and `pocket_center`.
- **Tool**: `ValidatePose`
  - Checks if the docked pose is geometrically valid (PoseBusters).
- **Tool**: `AnalyzeInteractions`
  - Identifies hydrogen bonds, hydrophobic contacts, etc.

### C. Filtering & Selection
- **Tool**: `FilterMolecules`
  - Applies hard cutoffs (e.g., QED > 0.5, Docking < -7.0).

## 4. Execution Workflow
1.  **Load Molecules**: Automatically loads generated molecules.
2.  **Initial Screen**: Run `RunStandardEvaluation` to get basic metrics.
3.  **Filter**: Discard molecules with poor properties (e.g., SA > 8, QED < 0.3).
4.  **Docking**: Run `RunDocking` on the survivors.
5.  **Final Selection**: Select best binders with good properties.
6.  **Finish**: Report success/failure and pass data to SynthesisAgent.
