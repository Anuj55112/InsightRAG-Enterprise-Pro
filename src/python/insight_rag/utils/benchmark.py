import os
import sys
import time
import json
import platform
import subprocess
from datetime import datetime

def get_system_metadata() -> dict:
    metadata = {
        "os": platform.system(),
        "cpu": "Unknown",
        "ram_gb": 8,
        "gpu": "None"
    }
    
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            metadata["cpu"] = cpu
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        metadata["cpu"] = line.split(":")[1].strip()
                        break
    except Exception:
        pass
        
    try:
        if platform.system() == "Darwin":
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
            metadata["ram_gb"] = round(mem_bytes / (1024 ** 3))
        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        metadata["ram_gb"] = round(mem_kb / (1024 ** 2))
                        break
    except Exception:
        pass
        
    try:
        import torch
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            metadata["gpu"] = "Apple Metal (MPS)"
    except Exception:
        pass
        
    return metadata

def run_benchmark():
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    report = {
        "project": "InsightRAG Pro",
        "timestamp": timestamp,
        "status": "not_run",
        "hardware": get_system_metadata(),
        "environment": {
            "python": platform.python_version()
        },
        "metadata": {
            "model": "NetworkX / all-MiniLM-L6-v2",
            "parameters_million": 22.7,
            "dataset": "Dynamic Knowledge Graph (100 entities)",
            "batch_size": 1,
            "image_size": None,
            "sequence_length": 600,
            "device": "cpu"
        },
        "benchmarks": {}
    }
    
    # Check dependencies
    required_libs = ["torch", "networkx", "sentence_transformers"]
    missing_deps = []
    
    for lib in required_libs:
        try:
            mod = __import__(lib)
            report["environment"][lib] = getattr(mod, "__version__", "installed")
        except ImportError:
            missing_deps.append(lib)
            
    if missing_deps:
        report["status"] = "not_run"
        report["reason"] = f"Missing required dependencies: {', '.join(missing_deps)}"
        report["required_dependency"] = missing_deps[0]
        
        save_reports(report, date_str)
        print(f"Benchmark not run: {report['reason']}")
        return
        
    try:
        from src.python.insight_rag.graph.knowledge_graph import KnowledgeGraph
        
        # Benchmarking NetworkX Graph Community modulations
        graph = KnowledgeGraph()
        
        # Add sample entities and relations
        for i in range(50):
            graph.add_relation(f"Entity_{i}", f"Relation_{i}", f"Entity_{i+1}")
            if i % 3 == 0:
                graph.add_relation(f"Entity_{i}", "core_relation", "Entity_Core")
                
        print("Benchmarking modularity-greedy community partition detection...")
        start_time = time.time()
        for _ in range(50):
            _ = graph.get_communities()
        graph_latency = ((time.time() - start_time) / 50) * 1000
        
        report["status"] = "success"
        report["benchmarks"] = {
            "community_partition_latency_ms": round(graph_latency, 2),
            "graph_nodes_count": len(graph.graph.nodes),
            "graph_edges_count": len(graph.graph.edges)
        }
        print(f"Benchmark success: GraphRAG community extraction on {len(graph.graph.nodes)} nodes = {graph_latency:.2f}ms")
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"Benchmark execution error: {e}"
        print(f"Benchmark error: {e}")
        
    save_reports(report, date_str)

def save_reports(report: dict, date_str: str):
    with open(f"reports/{date_str}-benchmark.json", "w") as f:
        json.dump(report, f, indent=4)
    with open("reports/latest.json", "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
