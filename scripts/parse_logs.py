import os
import re
import glob

def main():
    log_files = glob.glob("logs/eval-*-*.out")
    if not log_files:
        print("No log files found in logs/ directory.")
        return

    results = []

    for filepath in log_files:
        # Extract epoch number (e10, e20) or 'best' from the filename
        m_epoch = re.search(r'e(\d+)', filepath)
        is_best = 'best' in filepath
        
        if m_epoch:
            epoch = int(m_epoch.group(1))
            name = str(epoch)
        elif is_best:
            epoch = 999  # Sort 'best' at the end
            name = "Best"
        else:
            continue
        
        lp_mse, knn_mse = None, None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Use regex to find the MSE scores in the summary
                m_lp = re.search(r'Linear Probe — Total MSE:\s*([\d\.]+)', content)
                m_knn = re.search(r'kNN — Total MSE:\s*([\d\.]+)', content)
                
                if m_lp: lp_mse = float(m_lp.group(1))
                if m_knn: knn_mse = float(m_knn.group(1))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
            
        if lp_mse is not None and knn_mse is not None:
            results.append((epoch, name, lp_mse, knn_mse))

    # Sort results sequentially by epoch
    results.sort(key=lambda x: x[0])

    print("\n--- EVALUATION RESULTS ---\n")
    print("| Epoch | Linear Probe MSE | kNN MSE |")
    print("|-------|------------------|---------|")
    for r in results:
        print(f"| {r[1]:<5} | {r[2]:<16.4f} | {r[3]:<7.4f} |")

if __name__ == "__main__":
    main()
