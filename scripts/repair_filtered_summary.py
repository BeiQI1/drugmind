
import pandas as pd
import os
import argparse

def repair_filtered_summary(run_dir, num_samples=20):
    summary_path = os.path.join(run_dir, "benchmark_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    print(f"Reading full summary from {summary_path}...")
    df = pd.read_csv(summary_path)
    
    # Ensure Case_ID is present
    if "Case_ID" not in df.columns:
        print("Error: 'Case_ID' column missing in summary.")
        return

    all_filtered = []
    
    # Group by Case_ID to handle each target separately
    for case_id, group in df.groupby("Case_ID"):
        print(f"Processing Case {case_id}: Total {len(group)} molecules")
        
        # Sort logic: Qualified first, then by Score (Docking or QED)
        sort_cols = ["Qualified"]
        asc_order = [False] # True > False (True is 1, False is 0? No, boolean sort: False < True. So False=Ascending puts False first. We want True first, so Ascending=False)
        # Check: True > False is True in Python. sort_values(ascending=False) puts True first. Correct.
        
        if "Docking_Score" in group.columns:
            sort_cols.append("Docking_Score")
            asc_order.append(True) # Lower is better
        elif "QED" in group.columns:
            sort_cols.append("QED")
            asc_order.append(False) # Higher is better
            
        group_sorted = group.sort_values(by=sort_cols, ascending=asc_order)
        
        # Take top N
        top_n = group_sorted.head(num_samples).copy()
        all_filtered.append(top_n)

    if not all_filtered:
        print("No data found to filter.")
        return

    df_filtered = pd.concat(all_filtered, ignore_index=True)
    
    # Drop 'source' column if present
    if "source" in df_filtered.columns:
        df_filtered.drop(columns=["source"], inplace=True)
    
    output_path = os.path.join(run_dir, "benchmark_filtered_summary_repaired.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"Repaired filtered summary saved to {output_path}")
    print(f"Total rows: {len(df_filtered)}")
    
    # Validation for Case 19
    case_19 = df_filtered[df_filtered["Case_ID"] == 19]
    print(f"Validation Case 19 count: {len(case_19)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples per case")
    args = parser.parse_args()
    
    repair_filtered_summary(args.run_dir, args.num_samples)
