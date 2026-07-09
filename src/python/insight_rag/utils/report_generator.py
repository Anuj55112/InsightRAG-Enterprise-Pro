import os
import json

def generate_reports():
    latest_path = "reports/latest.json"
    if not os.path.exists(latest_path):
        print("No latest.json report found. Run benchmark first.")
        return
        
    with open(latest_path, "r") as f:
        report = json.load(f)
        
    md_content = f"""# Benchmark Verification Report: {report['project']}
- **Generated Timestamp**: {report['timestamp']}
- **Status**: {report['status'].upper()}

## Hardware Metadata
- **OS**: {report['hardware']['os']}
- **CPU**: {report['hardware']['cpu']}
- **RAM**: {report['hardware']['ram_gb']} GB
- **GPU**: {report['hardware']['gpu']}

## Environment Versions
- **Python**: {report['environment'].get('python', 'N/A')}
- **PyTorch**: {report['environment'].get('torch', 'N/A')}
- **NetworkX**: {report['environment'].get('networkx', 'N/A')}
- **SentenceTransformers**: {report['environment'].get('sentence_transformers', 'N/A')}

## Model Metadata
- **Model**: {report['metadata']['model']}
- **Dataset**: {report['metadata']['dataset']}
- **Device**: {report['metadata']['device']}

"""

    if report["status"] == "success":
        b = report["benchmarks"]
        md_content += f"""## Measured Benchmark Results
| Metric | Nodes Count | Edges Count | Modularity Latency |
| :--- | :---: | :---: | :---: |
| **GraphRAG Communities** | {b['graph_nodes_count']} nodes | {b['graph_edges_count']} edges | **{b['community_partition_latency_ms']} ms** |
"""
    else:
        md_content += f"""## Execution Note
- **Reason**: {report.get('reason', 'Unknown reason')}
"""

    with open("reports/latest.md", "w") as f:
        f.write(md_content)
    print("reports/latest.md updated successfully.")

    # 2. Update README.md
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            readme = f.read()
            
        start_marker = "<!-- BENCHMARK_TABLE_START -->"
        end_marker = "<!-- BENCHMARK_TABLE_END -->"
        
        if start_marker in readme and end_marker in readme:
            if report["status"] == "success":
                b = report["benchmarks"]
                table_md = f"""
| RAG Strategy | Context Scope | Graph Nodes / Edges | Latency |
| :--- | :---: | :---: | :---: |
| **Knowledge Graph Clusters** | Community-Greedy Modularity | {b['graph_nodes_count']} / {b['graph_edges_count']} | {b['community_partition_latency_ms']} ms |
"""
            else:
                table_md = f"\n*Benchmark Not Run: {report.get('reason', 'Missing dependencies')}*\n"
                
            start_idx = readme.find(start_marker) + len(start_marker)
            end_idx = readme.find(end_marker)
            
            new_readme = readme[:start_idx] + table_md + readme[end_idx:]
            with open(readme_path, "w") as f:
                f.write(new_readme)
            print("README.md benchmark table updated successfully.")
        else:
            print("Benchmark table markers not found in README.md.")

if __name__ == "__main__":
    generate_reports()
