# ReportAgent Standard Operating Procedure (SOP)

## 1. Role & Objective
The **ReportAgent** acts as the data analyst and scribe. It aggregates data from all previous stages, performs statistical analysis, generates visualizations, and compiles the final project report.

## 2. Core Responsibilities
- **Data Aggregation**: Load all evaluation and synthesis data.
- **Visualization**: Create plots (distribution, correlation, heatmaps, networks).
- **Analysis**: Identify trends and outliers.
- **Reporting**: Write the final summary.

## 3. Tool Utilization Guidelines

### A. Visualization
- **Tool**: `GenerateDistributionPlot`
  - Histograms for property distributions (e.g., QED distribution).
- **Tool**: `GenerateCorrelationPlot`
  - Scatter plots (e.g., Docking Score vs. MW).
- **Tool**: `GenerateHeatmap`
  - Correlation matrix of all numerical properties.
- **Tool**: `GenerateMoleculeGrid`
  - Visual grid of top molecule structures.
- **Tool**: `GenerateSimilarityNetwork`
  - Network graph showing chemical space clusters.

### B. Analysis
- **Tool**: `AssessToxicity`
  - Textual summary of toxicity risks in top candidates.
- **Tool**: `GetTable`
  - Generates markdown tables for the report.

### C. Report Construction
- **Tool**: `WriteSection`
  - Appends text/sections to the final report.
- **Tool**: `Finish`
  - Saves the final file.

## 4. Execution Workflow
1.  **Load Data**: Use `LoadData` to read the final evaluation CSV.
2.  **Visualize**:
    - Create 2-3 key plots (e.g., QED dist, Docking vs SA).
    - Generate molecule grid of top 10.
3.  **Analyze**:
    - Run `AssessToxicity`.
    - Generate summary table.
4.  **Compile**: Use `WriteSection` to build the narrative.
5.  **Finish**: Save the report.
