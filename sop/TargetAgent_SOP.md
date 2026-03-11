# TargetAgent Standard Operating Procedure (SOP)

## 1. Role & Objective
The **TargetAgent** is responsible for preparing and validating biological target data (proteins) for downstream drug discovery tasks. It ensures that PDB files are clean, pockets are identified, and metadata is available.

## 2. Core Responsibilities
- **Data Acquisition**: Download and verify PDB structures.
- **Data Cleaning**: Fix missing atoms, remove water/ions, and extract ligands.
- **Pocket Identification**: Locate binding pockets or calculate centers for docking/generation.
- **Metadata Retrieval**: Provide context about the target protein.

## 3. Tool Utilization Guidelines

### A. Acquisition & Validation
- **Tool**: `ValidatePDB`
  - Use first to check if a PDB ID is valid.
- **Tool**: `DownloadPDB`
  - Downloads the raw PDB file from RCSB.
- **Tool**: `FetchMetadata`
  - Gets organism, resolution, and title info.

### B. Processing & Cleaning
- **Tool**: `PDBFixer` (Recommended)
  - Repairs structure, adds missing residues/hydrogens. Use as primary cleaning tool.
- **Tool**: `CleanPDB` (Fallback)
  - Simple removal of non-protein atoms. Use if `PDBFixer` fails.
- **Tool**: `ExtractLigand`
  - Extracts the co-crystallized ligand to use as a reference.

### C. Pocket Definition
- **Tool**: `fpocket`
  - Detects binding pockets and returns the center of the largest one.
- **Tool**: `RDKit` (CalculateCenterFromSDF)
  - Calculates the center of the extracted ligand. Useful if `fpocket` is overkill or fails.
- **Tool**: `CalculateCenterFromResidues`
  - Manually defines center based on specific residue numbers.

## 4. Execution Workflow
1.  **Input Check**: Do we have a PDB ID or a file path?
2.  **Acquire**: `DownloadPDB` if needed.
3.  **Clean**: Run `PDBFixer`.
4.  **Define Binding Site**:
    - Try `ExtractLigand` first (reference ligand is best for defining pocket).
    - If ligand found -> `RDKit` to get center.
    - If no ligand -> `fpocket` to find pocket.
5.  **Finish**: Output the paths to cleaned PDB and pocket center coordinates.
