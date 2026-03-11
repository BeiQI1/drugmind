# SynthesisAgent Standard Operating Procedure (SOP)

## 1. Role & Objective
The **SynthesisAgent** plans how to synthesize the selected drug candidates. It performs retrosynthetic analysis to determine feasible routes from commercially available starting materials.

## 2. Core Responsibilities
- **Retrosynthesis Planning**: Deconstruct target molecules into available precursors.
- **Route Evaluation**: Assess the complexity and cost of synthesis routes.
- **Reporting**: Generate visual synthesis trees and step-by-step guides.

## 3. Tool Utilization Guidelines

### A. Retrosynthesis Engines
- **Tool**: `RunAiZynth` (Primary/Core)
  - Uses AiZynthFinder. Robust and reliable.
- **Tool**: `RunRetrosynthesis` (Secondary)
  - Uses Syntheseus. Good for alternative routes or cross-validation.

### B. Analysis & Reporting
- **Tool**: `AnalyzeRouteComplexity`
  - Counts steps and identifies starting materials.
- **Tool**: `GenerateSynthesisReport`
  - Compiles all findings into an HTML report with route images.

### C. Candidate Management
- **Tool**: `LoadCandidates`
  - Loads the molecules selected by EvaluatorAgent.
- **Tool**: `SelectTopN`
  - Picks the absolute best molecules if the list is still too long.

## 4. Execution Workflow
1.  **Load**: Get candidates from Evaluator results.
2.  **Plan**:
    - For each candidate, run `RunAiZynth`.
    - (Optional) Run `RunRetrosynthesis` for comparison.
3.  **Analyze**: Check if routes are found and how complex they are.
4.  **Report**: Generate the visual report.
5.  **Finish**: Output the report path.
