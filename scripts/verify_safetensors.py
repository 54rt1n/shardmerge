# verify_safetensors.py
# Copyright (C) 2025 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import json
import os
from collections import defaultdict
from safetensors import safe_open
import glob
import shutil
from datetime import datetime

def load_index(index_path):
    """Load the index file and return the weight map and full index."""
    with open(index_path) as f:
        index = json.load(f)
    return index['weight_map'], index

def get_actual_keys(safetensors_dir):
    """Read all safetensors files and return a mapping of files to their keys."""
    file_keys = {}
    for file in glob.glob(os.path.join(safetensors_dir, "*.safetensors")):
        model = safe_open(file, framework="pt")
        file_keys[os.path.basename(file)] = set(model.keys())
    return file_keys

def verify_alignment(weight_map, file_keys):
    """Verify alignment between index and actual files."""
    # Create reverse mapping from file to expected keys
    expected_file_keys = defaultdict(set)
    for key, file in weight_map.items():
        expected_file_keys[file].add(key)
    
    # Check for missing files
    missing_files = set(expected_file_keys.keys()) - set(file_keys.keys())
    if missing_files:
        print("Missing safetensors files:")
        for file in sorted(missing_files):
            print(f"  {file}")
        print()

    # Check for extra files
    extra_files = set(file_keys.keys()) - set(expected_file_keys.keys())
    if extra_files:
        print("Extra safetensors files not in index:")
        for file in sorted(extra_files):
            print(f"  {file}")
        print()

    # Check key mismatches for each file
    has_mismatches = False
    for file in sorted(set(expected_file_keys.keys()) & set(file_keys.keys())):
        expected = expected_file_keys[file]
        actual = file_keys[file]
        
        missing_keys = expected - actual
        extra_keys = actual - expected
        
        if missing_keys or extra_keys:
            has_mismatches = True
            print(f"Mismatches in {file}:")
            if missing_keys:
                print("  Missing keys (in index but not in file):")
                for key in sorted(missing_keys):
                    print(f"    {key}")
            if extra_keys:
                print("  Extra keys (in file but not in index):")
                for key in sorted(extra_keys):
                    print(f"    {key}")
            print()
    
    if not has_mismatches and not missing_files and not extra_files:
        print("All safetensors files align perfectly with the index!")
    
    return has_mismatches or missing_files or extra_files

def repair_index(file_keys, original_index, output_path):
    """Create a new index file based on actual files and their keys."""
    # Create new weight map
    new_weight_map = {}
    for file, keys in file_keys.items():
        for key in keys:
            new_weight_map[key] = file
    
    # Create new index
    new_index = {
        "metadata": original_index.get("metadata", {}),
        "weight_map": new_weight_map
    }
    
    # Backup original index if it exists
    if os.path.exists(output_path):
        backup_path = f"{output_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(output_path, backup_path)
        print(f"Backed up original index to: {backup_path}")
    
    # Write new index
    with open(output_path, 'w') as f:
        json.dump(new_index, f, indent=2)
    print(f"Wrote repaired index to: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify and repair safetensors files against index')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory containing the safetensors files')
    parser.add_argument('--index_path', type=str, required=True,
                      help='Path to the model.safetensors.index.json file')
    parser.add_argument('--repair', action='store_true',
                      help='If specified, create a new corrected index file')
    
    args = parser.parse_args()
    
    # Load index and get actual keys
    weight_map, original_index = load_index(args.index_path)
    file_keys = get_actual_keys(args.model_dir)
    
    # Verify alignment
    has_issues = verify_alignment(weight_map, file_keys)
    
    # Repair if requested and there are issues
    if args.repair:
        if has_issues:
            repair_index(file_keys, original_index, args.index_path)
        else:
            print("No repair needed - index is already correct!")

if __name__ == '__main__':
    main() 