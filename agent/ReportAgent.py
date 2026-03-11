import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from agent.base_agent import BaseAgent
from agent.state import AgentState
from agent.RAGAgent import RAGAgent
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs

class ReportAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="ReportAgent")
        # Create a timestamped subdirectory for each run to keep files organized
        self.run_id = f"report_{int(pd.Timestamp.now().timestamp())}"
        self.work_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reports", self.run_id)
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.df = None
        self.source_file_path = None
        self.report_content = []
        self.report_images = []
        self.attachment_files = []
        self.rag_agent = RAGAgent()
        
        # Tool implementations
        self.tool_implementations = {
            "LoadData": self.load_data,
            "RetrieveContext": self.retrieve_context,
            "GenerateDistributionPlot": self.generate_distribution_plot,
            "GenerateCorrelationPlot": self.generate_correlation_plot,
            "GenerateHeatmap": self.generate_heatmap,
            "GenerateMoleculeGrid": self.generate_molecule_grid,
            "GenerateSimilarityNetwork": self.generate_similarity_network,
            "GetTable": self.get_table,
            "AssessToxicity": self.assess_toxicity,
            "WriteSection": self.write_section,
            "Finish": None 
        }
        
        # Tool descriptions for the LLM
        self.tool_descriptions = {
            "LoadData": "Loads the evaluation CSV file. Args: file_path",
            "RetrieveContext": "Retrieves background info from KG/Literature. Args: query",
            "GenerateDistributionPlot": "Generates a histogram/KDE for a specific column. Args: column, title",
            "GenerateCorrelationPlot": "Generates a scatter plot between two columns. Args: x_column, y_column, title",
            "GenerateHeatmap": "Generates a correlation heatmap for numeric columns. Args: title",
            "GenerateMoleculeGrid": "Generates a grid image of top molecules. Args: sort_by (column, optional), top_k (int)",
            "GenerateSimilarityNetwork": "Generates a chemical similarity network of top molecules. Args: threshold (float, default 0.7), top_k (int, default 50)",
            "GetTable": "Returns a markdown table of the data. Args: columns (list of str, optional), top_k (int, optional), sort_by (str, optional)",
            "AssessToxicity": "Checks top molecules for potential toxicity risks. Args: top_k (int)",
            "WriteSection": "Adds a section to the report. Args: title, content (markdown)",
            "Finish": "Finalizes and saves the report. Args: filename"
        }

    def assess_toxicity(self, top_k: int = 10):
        if self.df is None:
            return "Error: No data loaded."
        
        try:
            # Respect original order (from EvaluatorAgent)
            top_df = self.df.head(top_k)
            
            warnings = []
            for idx, row in top_df.iterrows():
                # Handle rank: use existing if valid, else use index+1
                rank_val = row.get('rank')
                if pd.isna(rank_val):
                    mol_id = f"#{idx+1}"
                else:
                    mol_id = f"#{int(rank_val)}"
                    
                smiles = row.get('smiles', '')
                sa = row.get('SA', 0)
                qed = row.get('QED', 1)
                
                # Heuristic rules for "Toxicity/Risk"
                risks = []
                if sa > 4.5:
                    risks.append("High Synthetic Difficulty (SA > 4.5)")
                if qed < 0.4:
                    risks.append("Low Drug-likeness (QED < 0.4)")
                
                # Check for specific substructures (simple example)
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('[N+](=O)[O-]')): # Nitro group
                        risks.append("Contains Nitro group (Potential Toxicity Alert)")
                    if mol.HasSubstructMatch(Chem.MolFromSmarts('NN')): # Hydrazine
                        risks.append("Contains Hydrazine moiety (Potential Toxicity Alert)")
                
                if risks:
                    warnings.append(f"- **Molecule {mol_id}**: {', '.join(risks)}")
            
            if not warnings:
                return "No immediate structural alerts found in top candidates."
            
            report = "### Toxicity & Safety Assessment\n\nThe following top candidates exhibit structural alerts or properties indicating potential toxicity or development risks:\n\n" + "\n".join(warnings)
            report += "\n\n**Disclaimer**: This assessment is based on heuristic structural alerts and physicochemical properties. Experimental validation is required."
            return report
            
        except Exception as e:
            return f"Error assessing toxicity: {e}"

    def get_table(self, columns: List[str] = None, top_k: int = 10, sort_by: str = None):
        if self.df is None:
            return "Error: No data loaded."
        
        try:
            temp_df = self.df.copy()
            
            # IGNORE sort_by to respect EvaluatorAgent's MPO ranking
            # if sort_by:
            #     if sort_by in temp_df.columns:
            #         # Determine sort order
            #         ascending = True
            #         if "Docking" in sort_by or "Energy" in sort_by:
            #             ascending = True # Lower is better
            #         elif "QED" in sort_by:
            #             ascending = False # Higher is better
            #             
            #         temp_df = temp_df.sort_values(by=sort_by, ascending=ascending)
            #     else:
            #         return f"Error: Column {sort_by} not found."
            
            # Recalculate 'rank' for the report table to ensure it's clean (1 to N)
            # This replaces the original 'rank' which might have NaNs
            temp_df['rank'] = range(1, len(temp_df) + 1)
            
            if columns:
                requested_cols = list(columns)
                # Validate columns
                valid_cols = [c for c in requested_cols if c in temp_df.columns]
                if not valid_cols:
                    return "Error: No valid columns specified."
                temp_df = temp_df[valid_cols]
            
            if top_k:
                temp_df = temp_df.head(top_k)
                
            return temp_df.to_markdown(index=False)
        except Exception as e:
            return f"Error generating table: {e}"

    def load_data(self, file_path: str):
        if not os.path.exists(file_path):
            return f"Error: File {file_path} not found."
        try:
            self.source_file_path = file_path
            self.df = pd.read_csv(file_path)
            # Filter out unnamed columns
            self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
            columns = ", ".join(self.df.columns.tolist())
            stats = self.df.describe().to_string()
            return f"Data loaded successfully. Shape: {self.df.shape}. Columns: {columns}\nStatistics:\n{stats}"
        except Exception as e:
            return f"Error loading data: {e}"

    def retrieve_context(self, query: str):
        """Retrieves context using RAGAgent."""
        try:
            results = self.rag_agent.retrieve(query, k=3)
            return f"Retrieved Context:\n{results}"
        except Exception as e:
            return f"Error retrieving context: {e}"

    def generate_distribution_plot(self, column: str, title: str):
        if self.df is None:
            return "Error: No data loaded."
        if column not in self.df.columns:
            return f"Error: Column {column} not found."
            
        try:
            # Nature-style formatting
            with sns.axes_style("ticks"):
                plt.figure(figsize=(8, 6))
                
                # Nature-like color palette (NPG - Nature Publishing Group style)
                # Using a teal/green shade for bars and dark blue for KDE
                bar_color = "#00A087" 
                kde_color = "#3C5488"
                
                ax = sns.histplot(
                    data=self.df, 
                    x=column, 
                    kde=True, 
                    color=bar_color,
                    edgecolor="white",
                    linewidth=1.2,
                    alpha=0.8,
                    line_kws={'linewidth': 2.5, 'color': kde_color}
                )
                
                # Remove top and right spines for a cleaner look
                sns.despine(offset=10, trim=False)
                
                # Add minimal grid
                ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
                ax.xaxis.grid(False)
                
                # Typography
                plt.title(title, fontsize=14, fontweight='bold', pad=20, fontfamily='sans-serif')
                plt.xlabel(column, fontsize=12, fontfamily='sans-serif', fontweight='bold')
                plt.ylabel("Frequency", fontsize=12, fontfamily='sans-serif', fontweight='bold')
                
                # Tick styling
                plt.tick_params(axis='both', which='major', labelsize=10, width=1.5)
                
                plt.tight_layout()
                
                filename = f"dist_{column}_{int(pd.Timestamp.now().timestamp())}.png"
                filepath = os.path.join(self.work_dir, filename)
                plt.savefig(filepath, dpi=300, facecolor='white', transparent=False)
                plt.close()
            
            self.report_images.append(filepath)
            return f"Plot saved to {filepath}"
        except Exception as e:
            return f"Error generating plot: {e}"

    def generate_correlation_plot(self, x_column: str, y_column: str, title: str):
        if self.df is None:
            return "Error: No data loaded."
        if x_column not in self.df.columns or y_column not in self.df.columns:
            return f"Error: Columns {x_column} or {y_column} not found."
            
        try:
            with sns.axes_style("ticks"):
                plt.figure(figsize=(8, 6))
                
                # Nature style scatter
                sns.scatterplot(
                    data=self.df, 
                    x=x_column, 
                    y=y_column,
                    color="#E64B35", # NPG Red
                    s=100,
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.8
                )
                
                sns.despine(offset=10)
                plt.grid(True, linestyle='--', alpha=0.3)
                
                plt.title(title, fontsize=14, fontweight='bold', pad=20, fontfamily='sans-serif')
                plt.xlabel(x_column, fontsize=12, fontfamily='sans-serif', fontweight='bold')
                plt.ylabel(y_column, fontsize=12, fontfamily='sans-serif', fontweight='bold')
                
                plt.tight_layout()
                
                filename = f"corr_{x_column}_{y_column}_{int(pd.Timestamp.now().timestamp())}.png"
                filepath = os.path.join(self.work_dir, filename)
                plt.savefig(filepath, dpi=300, facecolor='white', transparent=False)
                plt.close()
            
            self.report_images.append(filepath)
            return f"Plot saved to {filepath}"
        except Exception as e:
            return f"Error generating plot: {e}"

    def generate_heatmap(self, title: str = "Correlation Heatmap"):
        if self.df is None:
            return "Error: No data loaded."
        
        try:
            preferred_columns = [
                "rank",
                "num_atoms",
                "QED",
                "SA",
                "MW",
                "LogP",
                "HBD",
                "HBA",
                "Docking_Score",
                "processing_time",
                "diversity",
                "Diversity_Score",
            ]

            selected_columns = [c for c in preferred_columns if c in self.df.columns]
            if selected_columns:
                numeric_df = self.df[selected_columns].apply(pd.to_numeric, errors="coerce")
            else:
                numeric_df = self.df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return "Error: No numeric columns for heatmap."
            usable_columns = [c for c in numeric_df.columns if numeric_df[c].notna().sum() >= 2]
            numeric_df = numeric_df[usable_columns]
            if numeric_df.shape[1] < 2:
                return "Error: Not enough numeric columns for heatmap."

            plt.figure(figsize=(10, 8))
            
            # Professional diverging palette
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            
            sns.heatmap(
                numeric_df.corr(min_periods=2),
                annot=True, 
                cmap=cmap, 
                fmt=".2f",
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": .8}
            )
            plt.title(title, fontsize=14, fontweight='bold', pad=20, fontfamily='sans-serif')
            plt.tight_layout()
            
            filename = f"heatmap_{int(pd.Timestamp.now().timestamp())}.png"
            filepath = os.path.join(self.work_dir, filename)
            plt.savefig(filepath, dpi=300, facecolor='white', transparent=False)
            plt.close()
            
            self.report_images.append(filepath)
            return f"Heatmap saved to {filepath}"
        except Exception as e:
            return f"Error generating heatmap: {e}"

    def generate_molecule_grid(self, sort_by: str = None, top_k: int = 6):
        if self.df is None:
            return "Error: No data loaded."
            
        try:
            # Force using the original order (EvaluatorAgent's MPO ranking)
            # We ignore the 'sort_by' parameter for SORTING purposes to respect the upstream ranking.
            try:
                top_k_int = int(top_k)
            except Exception:
                top_k_int = 6
            top_k_int = max(1, min(10, top_k_int))
            top_df = self.df.head(top_k_int)
            
            # However, we can use 'sort_by' to decide what value to display under the molecule
            if sort_by and sort_by in self.df.columns:
                display_label = sort_by
            else:
                # Default display label
                display_label = "Docking_Score" if "Docking_Score" in self.df.columns else "QED"

            mols = []
            legends = []
            rank_counter = 1
            for _, row in top_df.iterrows():
                smiles = row.get('smiles', '')
                if not isinstance(smiles, str) or not smiles:
                    continue
                    
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mols.append(mol)
                    val = row.get(display_label, 0)
                    # Format value
                    if isinstance(val, (int, float)):
                        val_str = f"{val:.2f}"
                    else:
                        val_str = str(val)
                        
                    legends.append(f"#{rank_counter} | {display_label}: {val_str}")
                rank_counter += 1
            
            if not mols:
                return "Error: No valid molecules found."
                
            img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300), legends=legends)
            
            filename = f"mol_grid_{sort_by if sort_by else 'top'}_{int(pd.Timestamp.now().timestamp())}.png"
            filepath = os.path.join(self.work_dir, filename)
            img.save(filepath)
            
            self.report_images.append(filepath)
            return f"Molecule grid saved to {filepath}"
        except Exception as e:
            return f"Error generating molecule grid: {e}"

    def generate_similarity_network(self, threshold: float = 0.7, top_k: int = 50):
        if self.df is None:
            return "Error: No data loaded."
        
        try:
            # Use original order (EvaluatorAgent's filtered list)
            subset_df = self.df.head(top_k).reset_index(drop=True)
            
            mols = [Chem.MolFromSmiles(s) for s in subset_df['smiles']]
            
            # Use new MorganGenerator API to avoid deprecation warning
            from rdkit.Chem import rdFingerprintGenerator
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fps = [mfpgen.GetFingerprint(m) for m in mols if m]
            
            if len(fps) < 2:
                return "Error: Not enough valid molecules for network."

            G = nx.Graph()
            for i in range(len(fps)):
                G.add_node(i, label=f"Mol{i}")
                
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    if sim > threshold:
                        G.add_edge(i, j, weight=sim)
            
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
            plt.title(f"Chemical Similarity Network (Threshold > {threshold})")
            
            filename = f"sim_network_{int(pd.Timestamp.now().timestamp())}.png"
            filepath = os.path.join(self.work_dir, filename)
            plt.savefig(filepath, facecolor='white', transparent=False)
            plt.close()
            
            self.report_images.append(filepath)
            return f"Similarity network saved to {filepath}"
        except Exception as e:
            return f"Error generating network: {e}"

    def write_section(self, title: str, content: str):
        section = f"## {title}\n\n{content}\n"
        self.report_content.append(section)
        return f"Section '{title}' added."

    def _sanitize_smiles_for_filename(self, smiles: str) -> str:
        if not isinstance(smiles, str):
            return ""
        return "".join([c if c.isalnum() else "_" for c in smiles])

    def _truncate_text(self, text: str, max_len: int = 160) -> str:
        if not isinstance(text, str):
            return ""
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 1)] + "…"

    def _get_route_state_score(self, route: Any) -> Optional[float]:
        if not isinstance(route, dict):
            return None
        scores = route.get("scores", {}) or {}
        if not isinstance(scores, dict):
            return None
        val = scores.get("state score")
        try:
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    def _extract_aizynth_steps_from_route(self, route: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(route, dict):
            return {"smiles": None, "steps": []}

        root_smiles = route.get("smiles")
        reactions: List[Dict[str, Any]] = []

        def traverse(node: Dict[str, Any]):
            if not isinstance(node, dict):
                return
            if node.get("type") != "mol":
                return

            product = node.get("smiles")
            children = node.get("children", []) or []
            for child in children:
                if not isinstance(child, dict) or child.get("type") != "reaction":
                    continue
                metadata = child.get("metadata", {}) or {}
                reactants: List[str] = []
                for grandchild in child.get("children", []) or []:
                    if isinstance(grandchild, dict) and grandchild.get("type") == "mol":
                        smi = grandchild.get("smiles")
                        if isinstance(smi, str) and smi:
                            reactants.append(smi)
                reactions.append(
                    {
                        "product": product,
                        "reactants": reactants,
                        "policy_name": metadata.get("policy_name"),
                        "template": metadata.get("template"),
                    }
                )
                for grandchild in child.get("children", []) or []:
                    traverse(grandchild)

        traverse(route)
        steps = list(reversed(reactions))
        return {"smiles": root_smiles, "steps": steps}

    def _extract_aizynth_steps(self, json_path: str) -> Dict[str, Any]:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            return {"smiles": None, "steps": []}

        if not isinstance(data, list) or not data:
            return {"smiles": None, "steps": []}

        route = data[0]
        if not isinstance(route, dict):
            return {"smiles": None, "steps": []}

        return self._extract_aizynth_steps_from_route(route)

    def _select_top_aizynth_routes(self, routes: List[Any], top_k: int = 2) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for idx, r in enumerate(routes):
            if not isinstance(r, dict):
                continue
            normalized.append({"idx": idx, "route": r, "state_score": self._get_route_state_score(r)})

        if not normalized:
            return []

        has_scores = any(isinstance(it.get("state_score"), float) for it in normalized)
        if has_scores:
            normalized.sort(key=lambda x: (x["state_score"] is not None, x["state_score"]), reverse=True)
        return normalized[: max(1, int(top_k))]

    def _generate_route_image_from_route(self, route: Dict[str, Any], img_path: str) -> bool:
        from rdkit.Chem import Draw, AllChem
        try:
            from PIL import Image
        except Exception:
            return False

        try:
            reactions: List[Any] = []

            def traverse(node: Any):
                if not isinstance(node, dict):
                    return
                if node.get("type") == "mol":
                    for child in node.get("children", []) or []:
                        if isinstance(child, dict) and child.get("type") == "reaction":
                            product = node.get("smiles")
                            reactants: List[str] = []
                            for grandchild in child.get("children", []) or []:
                                if isinstance(grandchild, dict) and grandchild.get("type") == "mol":
                                    smi = grandchild.get("smiles")
                                    if isinstance(smi, str) and smi:
                                        reactants.append(smi)
                            if isinstance(product, str) and product and reactants:
                                reactions.append((product, reactants))
                            for grandchild in child.get("children", []) or []:
                                traverse(grandchild)

            traverse(route)
            if not reactions:
                return False

            images: List[Any] = []
            for prod, reacts in reversed(reactions):
                try:
                    rs = ".".join(reacts)
                    rxn_smarts = f"{rs}>>{prod}"
                    rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
                    d2d = Draw.MolDraw2DCairo(600, 200)
                    d2d.DrawReaction(rxn)
                    d2d.FinishDrawing()
                    png_data = d2d.GetDrawingText()
                    import io

                    img = Image.open(io.BytesIO(png_data))
                    images.append(img)
                except Exception:
                    continue

            if not images:
                return False

            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            combined_img = Image.new("RGB", (max_width, total_height), (255, 255, 255))

            y_offset = 0
            for img in images:
                x_offset = (max_width - img.width) // 2
                combined_img.paste(img, (x_offset, y_offset))
                y_offset += img.height

            combined_img.save(img_path)
            self.report_images.append(img_path)
            return True
        except Exception:
            return False

    def _replace_markdown_section(self, markdown_text: str, section_title: str, replacement_section: str) -> str:
        import re

        if not isinstance(markdown_text, str):
            markdown_text = ""
        if not isinstance(replacement_section, str) or not replacement_section.strip():
            return markdown_text

        pattern = re.compile(
            rf"^##\s+{re.escape(section_title)}\s*$[\s\S]*?(?=^##\s+|\Z)",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(markdown_text))
        if matches:
            parts: List[str] = []
            last_end = 0
            for idx, m in enumerate(matches):
                parts.append(markdown_text[last_end:m.start()])
                if idx == 0:
                    parts.append(replacement_section.rstrip() + "\n\n")
                last_end = m.end()
            parts.append(markdown_text[last_end:])
            return "".join(parts).rstrip() + "\n\n"
        return markdown_text.rstrip() + "\n\n" + replacement_section.rstrip() + "\n\n"

    def _build_retrosynthesis_markdown(self, csv_path: Optional[str], synthesis_results: Dict[str, Any]) -> str:
        rank_by_smiles: Dict[str, int] = {}
        name_by_smiles: Dict[str, str] = {}
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                if "smiles" in df.columns and "rank" in df.columns:
                    for _, row in df.iterrows():
                        smi = row.get("smiles")
                        rank_val = row.get("rank")
                        if isinstance(smi, str) and smi:
                            if not pd.isna(rank_val):
                                try:
                                    rank_by_smiles[smi] = int(rank_val)
                                except Exception:
                                    pass
                            mol_name = row.get("name") if "name" in df.columns else None
                            if isinstance(mol_name, str) and mol_name:
                                name_by_smiles[smi] = mol_name
            except Exception:
                rank_by_smiles = {}
                name_by_smiles = {}

        route_data_files = synthesis_results.get("route_data_files", []) if isinstance(synthesis_results, dict) else []
        route_images = synthesis_results.get("route_images", []) if isinstance(synthesis_results, dict) else []

        route_image_basenames = {os.path.basename(p) for p in route_images if isinstance(p, str)}
        copied_images = {os.path.basename(p): p for p in self.report_images if isinstance(p, str)}
        available_basenames = set(route_image_basenames) | set(copied_images.keys())

        items: List[Dict[str, Any]] = []
        for json_path in route_data_files:
            if not isinstance(json_path, str) or not os.path.exists(json_path):
                continue
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = []

            if not isinstance(data, list) or not data:
                continue

            root_route = data[0] if isinstance(data[0], dict) else {}
            smi = root_route.get("smiles") if isinstance(root_route, dict) else None
            safe = self._sanitize_smiles_for_filename(smi) if isinstance(smi, str) else ""

            top_routes = self._select_top_aizynth_routes(data, top_k=2)
            routes_payload: List[Dict[str, Any]] = []
            for r_order, r in enumerate(top_routes, start=1):
                route_idx = r.get("idx")
                route_obj = r.get("route")
                score = r.get("state_score")
                extracted = self._extract_aizynth_steps_from_route(route_obj) if isinstance(route_obj, dict) else {"smiles": smi, "steps": []}
                steps = extracted.get("steps", []) or []

                img_basename = None
                if safe:
                    existing_basename = f"route_summary_{safe}.png"
                    if route_idx == 0 and existing_basename in available_basenames:
                        img_basename = existing_basename
                    else:
                        new_basename = f"route_summary_{safe}_route{r_order}.png"
                        dest_path = os.path.join(self.work_dir, new_basename)
                        if os.path.exists(dest_path):
                            img_basename = new_basename
                        else:
                            if isinstance(route_obj, dict) and self._generate_route_image_from_route(route_obj, dest_path):
                                img_basename = new_basename

                routes_payload.append(
                    {
                        "route_order": r_order,
                        "route_idx": route_idx,
                        "state_score": score,
                        "steps": steps,
                        "image": img_basename,
                    }
                )

            items.append({"smiles": smi, "routes": routes_payload})

        if not items and route_images:
            for img_path in route_images:
                if not isinstance(img_path, str):
                    continue
                items.append(
                    {
                        "smiles": None,
                        "routes": [
                            {
                                "route_order": 1,
                                "route_idx": None,
                                "state_score": None,
                                "steps": [],
                                "image": os.path.basename(img_path),
                            }
                        ],
                    }
                )

        if not items:
            return "## Retrosynthesis Routes\n\nNo retrosynthesis routes were generated.\n"

        lines: List[str] = []
        lines.append("## Retrosynthesis Routes")
        lines.append("")
        lines.append("The figures below show retrosynthesis routes generated by AiZynthFinder. Captions indicate the molecule rank from evaluation.")
        lines.append("")

        for idx, item in enumerate(items, start=1):
            smi = item.get("smiles")
            routes = item.get("routes", []) or []
            if not isinstance(routes, list):
                routes = []

            rank_val = rank_by_smiles.get(smi) if isinstance(smi, str) else None
            label = f"Rank {rank_val}" if isinstance(rank_val, int) else f"Molecule {idx}"

            display_name = name_by_smiles.get(smi) if isinstance(smi, str) else None
            if not display_name and isinstance(smi, str):
                display_name = smi

            lines.append(f"### {label}")
            lines.append("")
            lines.append("| Field | Value |")
            lines.append("|---|---|")
            if display_name:
                lines.append(f"| Molecule | `{display_name}` |")
            if isinstance(smi, str) and smi:
                lines.append(f"| SMILES | `{smi}` |")
            lines.append("")

            if routes:
                for route_payload in routes[:2]:
                    route_order = route_payload.get("route_order")
                    steps = route_payload.get("steps", []) or []
                    img_basename = route_payload.get("image")
                    score = route_payload.get("state_score")

                    lines.append(f"#### Route {route_order}")
                    lines.append("")
                    lines.append("| Metric | Value |")
                    lines.append("|---|---|")
                    if isinstance(score, float):
                        lines.append(f"| State score | {score:.4f} |")
                    else:
                        lines.append("| State score | - |")
                    lines.append(f"| Steps | {len(steps)} |")
                    lines.append("")
                    if img_basename:
                        lines.append(f"![{label} route {route_order}]({img_basename})")
                        lines.append("")

                    if steps:
                        lines.append("| Step | Reactants | Product | Policy |")
                        lines.append("|---:|---|---|---|")
                        for s_idx, step in enumerate(steps, start=1):
                            reactants = step.get("reactants", []) or []
                            product = step.get("product")
                            policy_name = step.get("policy_name")

                            if reactants:
                                reactants_str = " + ".join([f"`{r}`" for r in reactants])
                            else:
                                reactants_str = "`(unknown reactants)`"
                            product_str = f"`{product}`" if isinstance(product, str) and product else "`(unknown product)`"
                            policy_str = str(policy_name) if isinstance(policy_name, str) and policy_name else "-"
                            lines.append(f"| {s_idx} | {reactants_str} | {product_str} | {policy_str} |")
                        lines.append("")
                    else:
                        lines.append("No route steps available.")
                        lines.append("")
            else:
                lines.append("No routes available.")
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _construct_prompt(
        self,
        task_description: str,
        agent_logs: str = "",
        intent: str = "",
        has_csv: bool = True,
        available_columns: Optional[List[str]] = None,
        has_synthesis: bool = False,
    ) -> str:
        available_columns = available_columns or []
        has_docking = "Docking_Score" in available_columns

        if has_csv:
            tool_desc = self.tool_descriptions
        else:
            tool_desc = {k: v for k, v in self.tool_descriptions.items() if k in {"RetrieveContext", "WriteSection", "Finish"}}
        tools_str = "\n".join([f"- {name}: {desc}" for name, desc in tool_desc.items()])

        if has_csv:
            if intent == "evaluation":
                sop_context = f"""
0. **Mandatory First Step (Do this BEFORE any other tool calls)**:
   - IMMEDIATELY call `WriteSection` titled 'Introduction'.
   - IMMEDIATELY call `WriteSection` titled 'Workflow Traceability'.
   - Use the EXACT titles above (case-sensitive). Do not add punctuation like ':' or '(...)'.
   - Do NOT generate plots/tables/analysis before these two sections exist.
1. **Initialization**: Load the dataset using `LoadData`.
2. **Context Retrieval**:
   - Use `RetrieveContext` only if needed to explain the target or assay context.
3. **Introduction & Traceability**:
   - IMMEDIATELY use `WriteSection` titled 'Introduction' to briefly describe the report scope and goal.
   - IMMEDIATELY use `WriteSection` titled 'Workflow Traceability' summarizing only the agents that actually ran (especially Evaluation).
4. **Key Results**:
   - Use `GetTable` with columns that exist (prefer: `rank`, `smiles`, `QED`, `SA`, `MW`, `LogP`, `HBD`, `HBA`, and `Docking_Score` if present).
   - IMMEDIATELY use `WriteSection` titled 'Evaluation Summary' to present the table.
5. **Visual Analysis (Only If Applicable)**:
   - Generate `GenerateDistributionPlot` for key columns that exist.
   - If enough numeric columns exist, generate `GenerateHeatmap`.
   - IMMEDIATELY use `WriteSection` titled 'Property Analysis' to discuss and **EMBED** plots using `![Desc](path)`.
6. **Safety**:
   - Use `AssessToxicity` and write 'Safety Assessment'.
7. **Conclusion**:
   - Write a concise 'Conclusion' focused on whether candidates meet the requested criteria.
8. **Finalization**: Call `Finish` to save the report.
"""
            elif intent == "optimization":
                sop_context = f"""
0. **Mandatory First Step (Do this BEFORE any other tool calls)**:
   - IMMEDIATELY call `WriteSection` titled 'Introduction'.
   - IMMEDIATELY call `WriteSection` titled 'Workflow Traceability'.
   - Use the EXACT titles above (case-sensitive). Do not add punctuation like ':' or '(...)'.
   - Do NOT generate plots/tables/analysis before these two sections exist.
1. **Initialization**: Load the dataset using `LoadData`.
2. **Context Retrieval**:
   - Use `RetrieveContext` only if needed to explain the optimization goal.
3. **Introduction & Traceability**:
   - IMMEDIATELY use `WriteSection` titled 'Introduction' to briefly describe the optimization goal and report scope.
   - IMMEDIATELY use `WriteSection` titled 'Workflow Traceability' summarizing only the agents that actually ran (especially Generation + Evaluation).
4. **Optimization Outcome**:
   - Use `GetTable` with columns that exist (always include `rank` and `smiles` if present).
   - IMMEDIATELY use `WriteSection` titled 'Optimized Candidates' to present the table.
   - Generate `GenerateMoleculeGrid` for top candidates and embed it.
5. **Property Analysis**:
   - Generate `GenerateDistributionPlot` for key columns that exist (e.g., QED, SA, MW, and `Docking_Score` if present).
   - If enough numeric columns exist, generate `GenerateHeatmap`.
   - IMMEDIATELY use `WriteSection` titled 'Property Analysis' to discuss and **EMBED** plots.
6. **Safety**:
   - Use `AssessToxicity` and write 'Safety Assessment'.
7. **Conclusion**:
   - Write a concise 'Conclusion' focused on whether the optimization goal is achieved.
8. **Appendix**:
   - Use `GetTable` with `top_k=100` and columns that exist.
   - IMMEDIATELY use `WriteSection` titled 'Appendix: Full Dataset'.
9. **Finalization**: Call `Finish` to save the report.
"""
            else:
                sop_context = f"""
0. **Mandatory First Step (Do this BEFORE any other tool calls)**:
   - IMMEDIATELY call `WriteSection` titled 'Introduction'.
   - IMMEDIATELY call `WriteSection` titled 'Workflow Traceability'.
   - Use the EXACT titles above (case-sensitive). Do not add punctuation like ':' or '(...)'.
   - Do NOT generate plots/tables/analysis before these two sections exist.
1. **Initialization**: Load the dataset using `LoadData`.
2. **Context Retrieval**:
   - Check 'Agent Logs' for any pre-fetched 'Target Metadata' (from TargetAgent).
   - Use `RetrieveContext` to get *additional* background information if needed.
3. **Introduction & Traceability**:
   - IMMEDIATELY use `WriteSection` titled 'Introduction' using the Target Metadata (if available) and retrieved context.
   - IMMEDIATELY use `WriteSection` titled 'Workflow Traceability' summarizing the work of the agents found in the logs (e.g., Intent, Target, Coordinator, Generation, Evaluation, Synthesis). Explicitly summarize the coordination and decision-making process if a Coordinator or collaborative agent is present. Only include agents that actually ran.
4. **Analysis & Results (Interleaved)**:
   - **Correlations**:
     - Generate `GenerateHeatmap`.
     - Generate `GenerateCorrelationPlot` (MW vs LogP).
     - If `Docking_Score` exists, also generate `GenerateCorrelationPlot` (Docking vs SA).
     - IMMEDIATELY use `WriteSection` titled 'Correlation Analysis' to discuss these plots and **EMBED** them using `![Desc](path)`.
   - **Distributions**:
     - Generate `GenerateDistributionPlot` for QED, SA, MW (only for columns that exist).
     - If `Docking_Score` exists, also generate `GenerateDistributionPlot` for Docking Score.
     - IMMEDIATELY use `WriteSection` titled 'Property Distributions' to discuss these plots and **EMBED** them.
   - **Chemical Space**:
     - Generate `GenerateSimilarityNetwork`.
     - IMMEDIATELY use `WriteSection` titled 'Chemical Space Analysis' to discuss and **EMBED** the network image.
5. **Candidate Selection & Safety**:
   - Generate `GenerateMoleculeGrid` for top candidates.
   - Use `GetTable` with columns that actually exist in the dataset (always include `rank` and `smiles` if present).
   - IMMEDIATELY use `WriteSection` titled 'Top Candidates' to display the table and **EMBED** the molecule grid.
   - Use `AssessToxicity` to check for risks.
   - IMMEDIATELY use `WriteSection` titled 'Safety Assessment' to include the toxicity report.
   - **Retrosynthesis**:
     - If retrosynthesis images were generated (check logs), use `WriteSection` titled 'Retrosynthesis Routes' to display them. Embed images like `![Route](route_summary_....png)`.
6. **Conclusion**:
   - Write a 'Conclusion' section summarizing the findings.
7. **Appendix**:
   - Use `GetTable` with `top_k=100` and columns that exist in the dataset (prefer: `rank`, `smiles`, `QED`, `SA`, `MW`, `LogP`, `HBD`, `HBA`, and `Docking_Score` if present).
   - IMMEDIATELY use `WriteSection` titled 'Appendix: Full Dataset' to include this table.
8. **Finalization**: Call `Finish` to save the report.
"""
        else:
            sop_context = f"""
0. **Mandatory First Step (Do this BEFORE any other tool calls)**:
   - IMMEDIATELY call `WriteSection` titled 'Introduction'.
   - IMMEDIATELY call `WriteSection` titled 'Workflow Traceability'.
   - Use the EXACT titles above (case-sensitive). Do not add punctuation like ':' or '(...)'.
   - Do NOT generate retrosynthesis writeups before these two sections exist.
1. **Context Retrieval**:
   - Check 'Agent Logs' for any pre-fetched 'Target Metadata'.
   - Use `RetrieveContext` to get background information if needed.
2. **Introduction & Traceability**:
   - IMMEDIATELY use `WriteSection` titled 'Introduction' based on available context and the user's request.
   - IMMEDIATELY use `WriteSection` titled 'Workflow Traceability' summarizing only the agents that actually ran.
3. **Retrosynthesis Results**:
   - If retrosynthesis images are available (check logs), use `WriteSection` titled 'Retrosynthesis Routes' and embed them using `![Route](filename.png)`.
   - Mention that raw route JSON files are embedded as PDF attachments (if present).
4. **Conclusion**:
   - Write a short 'Conclusion' focused on retrosynthesis feasibility and next steps.
5. **Finalization**: Call `Finish` to save the report.
"""
        
        return f"""You are the ReportAgent, an expert scientific writer and data analyst. Your goal is to generate a high-quality, traceable engineering report.

Task: {task_description}

Report Context:
- Intent: {intent}
- Has CSV Data: {has_csv}
- Has Docking Column: {has_docking}
- Has Retrosynthesis Results: {has_synthesis}
- Available Columns: {available_columns}

Agent Logs (Use this for the 'Workflow Traceability' section):
{agent_logs}

Available Tools:
{tools_str}

Standard Operating Procedure (SOP):
{sop_context}

Important Rules:
1. Only summarize agents and steps that are present in the logs or data. Do not hallucinate missing steps.
2. If a specific plot cannot be generated (e.g., missing column), skip it and move to the next.
3. You MUST NOT call `Finish` unless sections 'Introduction' and 'Workflow Traceability' already exist. If missing, create them with `WriteSection` first.
4. Section titles must match exactly (case-sensitive): 'Introduction', 'Workflow Traceability', 'Safety Assessment', 'Retrosynthesis Routes', 'Conclusion'. Do not add punctuation.

Format your response as a JSON object with the following structure:
{{
    "thought": "Your reasoning here",
    "tool": "ToolName",
    "args": { "arg_name": "value" }
}}
"""

    def run(self, state: AgentState) -> Dict[str, Any]:
        print(f"\n[{self.agent_name}] Starting report generation...")
        
        # Handle Synthesis Results
        synthesis_results = state.get("results", {}).get("synthesis", {})
        added_logs = ""
        if isinstance(synthesis_results, dict):
            # Images
            route_images = synthesis_results.get("route_images", [])
            for img_path in route_images:
                if os.path.exists(img_path):
                    import shutil
                    basename = os.path.basename(img_path)
                    dest_path = os.path.join(self.work_dir, basename)
                    shutil.copy2(img_path, dest_path)
                    self.report_images.append(dest_path)
                    added_logs += f"\n- Retrosynthesis Image Available: {basename}"
            
            # Data Files (Attachments)
            route_data = synthesis_results.get("route_data_files", [])
            for file_path in route_data:
                if os.path.exists(file_path):
                    self.attachment_files.append(file_path)
                    added_logs += f"\n- Retrosynthesis Data Attachment: {os.path.basename(file_path)}"
        
        task_params = state.get("task_params", {})
        csv_path = task_params.get("csv_path")
        intent = state.get("intent", "") or task_params.get("intent", "")
        has_csv = bool(csv_path and os.path.exists(csv_path))
        agent_logs = task_params.get("agent_logs", "No specific agent logs provided. Infer workflow from standard DiffSBDD pipeline.")
        agent_logs += added_logs
        
        has_synthesis = isinstance(synthesis_results, dict) and (
            bool(synthesis_results.get("route_images")) or bool(synthesis_results.get("route_data_files"))
        )
        if not has_csv and not has_synthesis:
            print(f"[{self.agent_name}] No CSV and no retrosynthesis results provided. Skipping report generation.")
            return {"error": "No CSV and no retrosynthesis results provided"}

        available_columns: List[str] = []
        if has_csv:
            try:
                df_head = pd.read_csv(csv_path, nrows=5)
                df_head = df_head.loc[:, ~df_head.columns.str.contains('^Unnamed')]
                available_columns = df_head.columns.tolist()
            except Exception:
                available_columns = []
        
        messages = [SystemMessage(content=self.get_system_prompt())]
        task_desc = f"Generate a comprehensive report for {csv_path}" if has_csv else "Generate a retrosynthesis-focused report without tabular evaluation data"
        messages.append(
            HumanMessage(
                content=self._construct_prompt(
                    task_desc,
                    agent_logs,
                    intent=intent,
                    has_csv=has_csv,
                    available_columns=available_columns,
                    has_synthesis=has_synthesis,
                )
            )
        )
        
        max_steps = 25
        current_step = 0
        
        while current_step < max_steps:
            response = self.model.invoke(messages)
            content = response.content
            
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                action_data = json.loads(content)
                thought = action_data.get("thought")
                action = action_data.get("tool")
                args = action_data.get("args", {})
                
                print(f"[{self.agent_name}] Step {current_step+1}: {thought}")
                print(f"[{self.agent_name}] Action: {action}")
                
                if action == "Finish":
                    filename = args.get("filename", "report.md")
                    # Ensure filename is just a basename to prevent directory traversal
                    filename = os.path.basename(filename)
                    
                    # Ensure filename ends with .md for consistency and to trigger PDF generation
                    if not filename.endswith(".md"):
                        filename += ".md"
                        
                    full_path = os.path.join(self.work_dir, filename)
                    
                    final_content = f"# Drug Discovery Experiment Report\n\n**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                    final_content += "\n".join(self.report_content)

                    if has_synthesis and isinstance(synthesis_results, dict):
                        retro_md = self._build_retrosynthesis_markdown(csv_path if has_csv else None, synthesis_results)
                        final_content = self._replace_markdown_section(final_content, "Retrosynthesis Routes", retro_md)
                    
                    # Only append images that were NOT used in the report content
                    unused_images = [img for img in self.report_images if os.path.basename(img) not in final_content]
                    if unused_images:
                        final_content += "\n\n## Appendix: Additional Figures\n"
                        for img_path in unused_images:
                            rel_path = os.path.basename(img_path)
                            final_content += f"\n![{rel_path}]({rel_path})\n"

                    # Add note about attachment if source file exists
                    if self.source_file_path and os.path.exists(self.source_file_path):
                        final_content += "\n\n## Data Attachment\n"
                        final_content += "The original CSV dataset used for this analysis is embedded as an attachment to this PDF file. "
                        final_content += "Please check the **Attachments** panel in your PDF viewer to access the raw data."

                    with open(full_path, "w") as f:
                        f.write(final_content)
                        
                    # Convert to PDF if requested or by default
                    if filename.endswith(".md"):
                        pdf_path = full_path.replace(".md", ".pdf")
                        
                        import markdown
                        import re
                        
                        # FIX: Sanitize image paths in markdown before conversion
                        # Ensure all image links point to just the filename, as base_url will handle the directory
                        def replace_path(match):
                            alt_text = match.group(1)
                            full_path = match.group(2)
                            filename = os.path.basename(full_path)
                            # Create a figure wrapper with caption
                            # IMPORTANT: No indentation in the returned string to avoid Markdown code block interpretation
                            return f'''<figure style="text-align: center; margin: 20px 0;">
<img src="{filename}" alt="{alt_text}" style="max-width: 100%; height: auto; border: 1px solid #ddd; padding: 5px; background: white;">
<figcaption style="text-align: center; font-style: italic; font-size: 9pt; color: #555; margin-top: 5px;">Figure: {alt_text}</figcaption>
</figure>'''

                        # Regex to match image links: ![AltText](Path)
                        final_content_fixed = re.sub(r'!\[(.+)\]\(([^)]+)\)', replace_path, final_content)
                        
                        # Convert Markdown to HTML
                        html_body = markdown.markdown(final_content_fixed, extensions=['tables', 'fenced_code'])
                        
                        # Add professional styling
                        html_text = f"""
                        <html>
                        <head>
                        <style>
                            @page {{
                                size: A4;
                                margin: 2cm;
                                @frame footer_frame {{
                                    -pdf-frame-content: footerContent;
                                    bottom: 1cm;
                                    margin-left: 1cm;
                                    margin-right: 1cm;
                                    height: 1cm;
                                }}
                            }}
                            body {{ 
                                font-family: "Helvetica", "Arial", sans-serif; 
                                font-size: 10pt; 
                                line-height: 1.4; 
                                color: #333;
                            }}
                            h1 {{ 
                                color: #1a5f7a; 
                                font-size: 22pt; 
                                border-bottom: 2px solid #1a5f7a; 
                                padding-bottom: 10px; 
                                margin-bottom: 20px;
                            }}
                            h2 {{ 
                                color: #2c3e50; 
                                font-size: 16pt; 
                                margin-top: 20px; 
                                margin-bottom: 10px;
                                border-left: 5px solid #1a5f7a;
                                padding-left: 10px;
                                background-color: #f0f4f8;
                                padding: 5px 10px;
                            }}
                            h3 {{ 
                                color: #34495e; 
                                font-size: 12pt; 
                                margin-top: 15px; 
                                font-weight: bold;
                            }}
                            p {{ 
                                margin-bottom: 10px; 
                                text-align: justify;
                            }}
                            img {{ 
                                max-width: 100%; 
                                height: auto; 
                                margin: 15px auto; 
                                display: block;
                                border: 1px solid #ddd;
                                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                            }}
                            /* Ensure images are treated as block elements for proper spacing */
                            figure {{
                                margin: 0;
                                padding: 0;
                                text-align: center;
                            }}
                            table {{ 
                                border-collapse: collapse; 
                                width: 100%; 
                                margin: 20px 0; 
                                font-size: 8pt; /* Smaller font for tables to prevent squeezing */
                                table-layout: fixed;
                                word-wrap: break-word;
                            }}
                            th, td {{ 
                                border: 1px solid #e0e0e0; 
                                padding: 6px; 
                                text-align: left; 
                                vertical-align: top;
                                word-break: break-all;
                            }}
                            th {{ 
                                background-color: #1a5f7a; 
                                color: #ffffff; 
                                font-weight: bold; 
                            }}
                            tr:nth-child(even) {{ 
                                background-color: #f9f9f9; 
                            }}
                        </style>
                        </head>
                        <body>
                        {html_body}
                        <div id="footerContent" style="text-align:center; font-size: 8pt; color: #777;">
                            Generated by DrugToolAgent - Page <pdf:pagenumber>
                        </div>
                        </body>
                        </html>
                        """

                        try:
                            # Try WeasyPrint first (Better quality)
                            from weasyprint import HTML, Attachment
                            
                            # Prepare attachments
                            pdf_attachments = []
                            if self.source_file_path and os.path.exists(self.source_file_path):
                                pdf_attachments.append(Attachment(self.source_file_path, description="Source Data CSV"))
                            
                            # Add Synthesis Attachments
                            for attach_path in self.attachment_files:
                                if os.path.exists(attach_path):
                                    desc = f"Retrosynthesis Data ({os.path.basename(attach_path)})"
                                    pdf_attachments.append(Attachment(attach_path, description=desc))
                            
                            # Ensure base_url is absolute path to work_dir for image resolution
                            base_url = f"file://{os.path.abspath(self.work_dir)}/"
                            
                            HTML(string=html_text, base_url=base_url).write_pdf(pdf_path, attachments=pdf_attachments)
                            print(f"[{self.agent_name}] PDF report generated with WeasyPrint at {pdf_path} (with {len(pdf_attachments)} attachments)")
                            full_path = pdf_path
                        except Exception as e:
                            print(f"[{self.agent_name}] WeasyPrint failed: {e}. Trying fallback to xhtml2pdf...")
                            # Fallback to xhtml2pdf
                            try:
                                from xhtml2pdf import pisa
                                
                                # Helper function to resolve paths for xhtml2pdf
                                def link_callback(uri, rel):
                                    sUrl = uri
                                    if not sUrl.startswith("http"):
                                        if not os.path.isabs(sUrl):
                                            sUrl = os.path.join(self.work_dir, sUrl)
                                        if not os.path.isfile(sUrl):
                                            print(f"Warning: Image not found at {sUrl}")
                                            return None
                                    return sUrl

                                with open(pdf_path, "wb") as pdf_file:
                                    pisa_status = pisa.CreatePDF(
                                        html_text, 
                                        dest=pdf_file,
                                        link_callback=link_callback
                                    )
                                    
                                if not pisa_status.err:
                                    print(f"[{self.agent_name}] PDF report generated with xhtml2pdf at {pdf_path}")
                                    full_path = pdf_path
                                else:
                                    print(f"[{self.agent_name}] PDF generation error.")
                            except ImportError:
                                print(f"[{self.agent_name}] PDF generation skipped (neither weasyprint nor xhtml2pdf installed).")
                            except Exception as e:
                                print(f"[{self.agent_name}] PDF generation failed: {e}")
                                import traceback
                                traceback.print_exc()

                    return {
                        "messages": [AIMessage(content=f"Report generated successfully at {full_path}")],
                        "report_path": full_path
                    }
                
                if action in self.tool_implementations:
                    tool_func = self.tool_implementations[action]
                    try:
                        result = tool_func(**args)
                    except Exception as e:
                        result = f"Error executing {action}: {str(e)}"
                else:
                    result = f"Error: Tool '{action}' not found."
                
                print(f"[{self.agent_name}] Result: {str(result)[:100]}...")
                
                messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content=f"Tool Output: {result}"))
                
            except json.JSONDecodeError:
                print(f"[{self.agent_name}] JSON Parse Error. Retrying...")
                messages.append(HumanMessage(content="Error: Invalid JSON format. Please respond with valid JSON."))
            except Exception as e:
                print(f"[{self.agent_name}] Error: {e}")
                break
                
            current_step += 1
            
        return {"messages": [AIMessage(content="Report generation timed out.")]}

# LangGraph Node Wrapper
def report_agent_node(state: AgentState) -> Dict[str, Any]:
    agent = ReportAgent()
    return agent.run(state)
