import os
import torch
from matplotlib import pyplot as plt
def load_all_tensors(out_dir="tensor_logs", max_files=None):
    files = sorted([f for f in os.listdir(out_dir) if f.endswith(".pt")])
    if max_files:
        files = files[:max_files]
    
    groups = []
    for f in files:
        data = torch.load(os.path.join(out_dir, f))
        for k in data:
            data[k] = data[k].squeeze(0)
        groups.append(data)
    return groups

def stats_all(groups):
    max_len = max(g["response mask"].numel() for g in groups)

    avg_rollout, min_rollout = [], []
    avg_ratio_cur_rollout, min_ratio_cur_rollout = [], []
    avg_ratio_cur_old = []

    for pos in range(max_len):
        vals_rollout = []
        vals_ratio_cur_rollout = []
        vals_ratio_cur_old = []

        for g in groups:
            mask = g["response mask"].view(-1)
            if pos < mask.numel() and mask[pos].item() > 0:  # 只算有效部分
                r = g["rollout logits"][pos]
                c = g["current logits"][pos]
                o = g["old_log_prob"][pos]

                vals_rollout.append(torch.exp(r).item())
                if abs(r.item()) > 1e-12:
                    vals_ratio_cur_rollout.append((c / r).item())
                if abs(o.item()) > 1e-12:
                    vals_ratio_cur_old.append((c / o).item())

        avg_rollout.append(sum(vals_rollout) / len(vals_rollout) if vals_rollout else float("nan"))
        avg_ratio_cur_rollout.append(sum(vals_ratio_cur_rollout) / len(vals_ratio_cur_rollout) if vals_ratio_cur_rollout else float("nan"))
        avg_ratio_cur_old.append(sum(vals_ratio_cur_old) / len(vals_ratio_cur_old) if vals_ratio_cur_old else float("nan"))
                # 最小值
        min_rollout.append(min(vals_rollout) if vals_rollout else float("nan"))
        min_ratio_cur_rollout.append(min(vals_ratio_cur_rollout) if vals_ratio_cur_rollout else float("nan"))

    return {
        "rollout_avg": torch.tensor(avg_rollout),
        "rollout_min": torch.tensor(min_rollout),
        "cur_over_rollout_avg": torch.tensor(avg_ratio_cur_rollout),
        "cur_over_rollout_min": torch.tensor(min_ratio_cur_rollout),
        "cur_over_old_avg": torch.tensor(avg_ratio_cur_old),
    }


# 使用示例
if __name__ == "__main__":
    groups = load_all_tensors("../tensor_logs", max_files=200)    
    results = stats_all(groups)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 第一个子图：rollout avg
    axes[0].plot(results["rollout_avg"], "g-", label="exp(rollout logits) avg")
    axes[0].set_ylabel("exp(rollout logits)")
    axes[0].grid(True)
    axes[0].legend()

    # 第二个子图：current/rollout avg
    axes[1].plot(results["cur_over_rollout_avg"], "b-", label="current/rollout avg")
    axes[1].set_ylabel("current / rollout")
    axes[1].grid(True)
    axes[1].legend()

    # 第三个子图：两个 min + current/old avg
    axes[2].plot(results["rollout_min"], "g--", label="rollout min")
    axes[2].plot(results["cur_over_rollout_min"], "b--", label="current/rollout min")
    axes[2].plot(results["cur_over_old_avg"], "r-", label="current/old avg")
    axes[2].set_ylabel("min / avg comparison")
    axes[2].grid(True)
    axes[2].legend()

    axes[2].set_xlabel("Position")

    fig.suptitle("Averages and Minimums across groups (organized)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("average_min_plot_organized.png")
    plt.show()


