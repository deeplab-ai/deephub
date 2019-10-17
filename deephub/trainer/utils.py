from subprocess import Popen, PIPE
from typing import List


def find_idle_gpu_devices(max_gpu_usage: int = 5, max_mem_usage: int = 5) -> List[int]:
    """
    Find all idle GPUs in the operating environment bases on their resources usages

    :param max_gpu_usage: The maximum processing usage a gpu must have to consider available
    :param max_mem_usage: The maximum memory usage a gpu must have to consider available
    :rtype: List[int]
    :return: The ids of the available gpu devices
    """
    query_fields = ["index", "utilization.gpu", "memory.total", "memory.used", "memory.free"]

    p = Popen(["nvidia-smi", "--query-gpu=" + ",".join(query_fields), "--format=csv,noheader,nounits"], stdout=PIPE)
    output = p.stdout.read().decode('UTF-8').strip()

    # Parse in 2D array of integers
    rows = map(lambda row: map(int, row.split(',')), output.split('\n'))

    # Convert to rows of dictionary mapped by field names
    rows = [
        dict(zip(query_fields, row))
        for row in rows
    ]

    available_devices = [
        row['index']
        for row in rows
        if row['utilization.gpu'] < max_gpu_usage and row['memory.used'] < max_mem_usage
    ]
    return available_devices
