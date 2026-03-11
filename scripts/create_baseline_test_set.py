import argparse
import sys
import os
from pyuul import utils
import torch
from tqdm import tqdm

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try importing constants
try:
    import dataset.crossdocked2020.constants as constants_module
    ELEMENT_2_HASH_FUNCBIND = constants_module.ELEMENT_2_HASH_FUNCBIND
except ImportError:
    # Fallback if module structure is different
    # Define standard element hash if needed or warn
    print("Warning: Could not import constants_module. Using default hashing if available.")
    # Assuming utils.atomlistToChannels can handle default or we need to define it
    # Let's try to define a minimal one if needed, but likely pyuul has defaults.
    # Actually, looking at the code, it's passed explicitly.
    # If we can't find it, we might fail.
    # Let's assume the path fix above works.
    pass

def has_structure(data_dir: str, poc: str, fname="crossdocked_pocket10") -> bool:
    target_pdb = os.path.join(data_dir, fname, poc)
    if not os.path.exists(target_pdb):
        return False
    try:
        coords, _ = utils.parsePDB(target_pdb)
        if coords[0].shape[0] == 0:
            return False
        return True
    except:
        return False

def preprocess_split(data: list, data_dir: str, fname="crossdocked_pocket10", rep_type='all_atom') -> list:
    processed_data = []

    for i, (poc, lig) in tqdm(enumerate(data), total=len(data)):
        # print(f">> processing sample {i + 1}/{len(data)}, pocket: {poc}, ligand: {lig}")

        if not has_structure(data_dir, poc, fname):
            # print(">> ignore pocket id:", poc)
            continue

        # Construct paths
        # Note: poc and lig are relative paths like "subdir/file.pdb"
        # We want to store relative paths in the output as requested by user
        # "采用相对路径"
        
        # Pocket
        pocket_pdb = os.path.join(data_dir, fname, poc)
        coords, atname = utils.parsePDB(pocket_pdb)
        atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENT_2_HASH_FUNCBIND)
        
        # Store relative path in a way that matches the old format if possible, 
        # or just "data/crossdocked_pocket10/..." 
        # The user wants "data/mcpp_dataset/..." but that folder doesn't exist.
        # Maybe they want "data/crossdocked_pocket10/..."?
        # Let's use the actual relative path from project root if possible.
        # If the baseline expects "data/mcpp_dataset", we might need to fake it or use the real one.
        # Given "采用相对路径" (use relative paths), I will use the relative path from the dataset root.
        
        # Actually, let's look at the inspect output again.
        # The old data had: 'prot_paths': ['data/mcpp_dataset/1bm2/1bm2-CP.pdb', ...]
        # This looks like it expects a `data` folder in the root.
        # Our data is in `test/dataset/crossdocked2020/crossdocked_pocket10`.
        # I will store the path relative to `drugtoolagent` root: `test/dataset/crossdocked2020/crossdocked_pocket10/...`
        
        rel_pocket_path = os.path.join("test", "dataset", "crossdocked2020", fname, poc)
        
        pocket = {
            "id": poc,
            "coords": coords[0].clone(),
            "atoms_channel": atoms_channel[0].type(torch.uint8),
            # Add path info for consistency check if needed
            "prot_paths": [rel_pocket_path] # Using list to match old format
        }

        # Ligand
        ligand_sdf = os.path.join(data_dir, fname, lig)
        coords, atname = utils.parseSDF(ligand_sdf)
        atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENT_2_HASH_FUNCBIND)
        c_min, _ = coords[0].min(axis=0)
        c_max, _ = coords[0].max(axis=0)
        
        ligand = {
            "id": lig,
            "coords": coords[0].clone(),
            "atoms_channel": atoms_channel[0].type(torch.uint8),
            "max_len": round((c_max - c_min).max().item(), 2),
        }

        # The old format seemed to merge them?
        # inspect_test_data.py showed:
        # { 'pos': ..., 'atoms_channel': ..., 'prot_paths': ... }
        # It seems it was a single dict per sample, not a tuple of (pocket, ligand).
        # But `preprocess_crossdocked.py` returns `(pocket, ligand)`.
        # The user's `inspect_test_data.py` showed a LIST of DICTS.
        # Each dict had `coords`, `atoms_channel`, `prot_paths`.
        # This looks like the POCKET data, but maybe with ligand info?
        # Wait, the old data had `prot_paths` pointing to PDBs.
        # And `coords`.
        
        # If the user wants the exact old format, I should follow `inspect_test_data.py` structure.
        # Old structure:
        # {
        #   'pos': tensor(...),          # Likely pocket coords? Or Complex?
        #   'atoms_channel': tensor(...),
        #   'aa': tensor(...),           # Amino acids? (Not in preprocess_crossdocked.py)
        #   'prot_paths': [path1, path2]
        # }
        
        # `preprocess_crossdocked.py` produces `(pocket, ligand)` tuples.
        # This is DIFFERENT from `inspect_test_data.py` output.
        # The user said "我的crossdocked数据集中本身有split_by_name.pt... 重新编写一个构建测试集代码".
        # It implies using `preprocess_crossdocked.py` logic is correct for *this* dataset.
        # The `test_data.pt` I inspected initially might have been from a DIFFERENT project/codebase ("测试集是我直接从别的项目下载的").
        
        # So I should generate what `preprocess_crossdocked.py` generates, but just for the test split.
        # And ensure paths are correct.
        
        processed_data.append((pocket, ligand))
        
    return processed_data

if __name__ == "__main__":
    base_dir = os.path.join(project_root, "test", "dataset", "crossdocked2020")
    fname = "crossdocked_pocket10"
    split_file = os.path.join(base_dir, "split_by_name.pt")
    output_path = os.path.join(project_root, "test", "dataset", "baseline_test_data.pt")
    
    print(f"Loading split from: {split_file}")
    split_by_name = torch.load(split_file)
    test_split = split_by_name['test']
    
    print(f"Processing {len(test_split)} samples...")
    data = preprocess_split(test_split, base_dir, fname=fname)
    
    print(f"Saving {len(data)} samples to {output_path}...")
    torch.save(data, output_path)
    print("Done.")
