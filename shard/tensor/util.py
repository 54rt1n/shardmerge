# shard/tensor/util.py
# Copyright (C) 2024 Martin Bukowski
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

import psutil
import contextlib
import torch

@contextlib.contextmanager
def cuda_memory_profiler(devices : list[str] = ['cuda:0'], display : str = True, title : str = "CUDA Memory Usage"):
    """
    A context manager for profiling CUDA memory usage in PyTorch.
    """
    if display is False:
        yield
        return
    
    # Get initial system RAM usage
    system_start = psutil.Process().memory_info().rss
     
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        starts = []
        for device in devices:
            starts.append(torch.cuda.memory_allocated(device))

    try:
        yield
    finally:
        # Get final system RAM usage
        system_end = psutil.Process().memory_info().rss
        
        print("\n" + title)
        # Print system RAM stats
        print("\nSystem Memory:")
        print(f"Start RAM usage: {system_start / (1024 ** 2):.2f} MB")
        print(f"End RAM usage: {system_end / (1024 ** 2):.2f} MB")
        print(f"Net RAM change: {(system_end - system_start) / (1024 ** 2):.2f} MB")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            ends = []
            for device in devices:
                ends.append(torch.cuda.memory_allocated(device))
            for i in range(len(starts)):
                start_memory = starts[i]
                end_memory = ends[i]
                print(f"Device: {devices[i]}")
                print(f"Peak memory usage: {torch.cuda.max_memory_allocated(devices[i]) / (1024 ** 2):.2f} MB")
                print(f"Memory allocated at start: {start_memory / (1024 ** 2):.2f} MB")
                print(f"Memory allocated at end: {end_memory / (1024 ** 2):.2f} MB")
                print(f"Net memory change: {(end_memory - start_memory) / (1024 ** 2):.2f} MB")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
