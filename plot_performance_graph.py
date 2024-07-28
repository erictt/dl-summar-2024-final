import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    "reduction_factor": [32, 16, 8, 4],
    "MH": [78.48453790052511, 80.0521476761068, 81.52592408783839, 81.91815525007279],
    "Output": [78.56006070686142, 80.31803632386513, 81.26012946094882, 81.95067804161363],
    "Double": [80.16030422326828, 81.22953410085692, 82.20740610600804, 82.79272097494145],
    "Double + Inv": [80.41014578996885, 81.24862350436496, 82.32785022398171, 82.86460658011505],
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(14, 6))

# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
colors = {"MH": "#FF7F0E", "Output": "#8ED8FF", "Double": "#FF8ED2", "Double + Inv": "#B888E6"}
adapter_types = colors.keys()

bar_width = 0.2
x = np.arange(len(df["reduction_factor"]))
for i, adapter in enumerate(adapter_types):
    ax.bar(x + i * bar_width, df[adapter], width=bar_width, label=adapter, color=colors[adapter])

ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(df["reduction_factor"])

# baseline
ax.axhline(y=82.91251169582613, color="r", linestyle="--", label="deepset/roberta-base-squad2")

ax.set_xlabel("Reduction Factor")
ax.set_ylabel("F1 Score")
ax.set_title("Adapter Training Results")

ax.legend(title="Adapter Type", loc="lower right")

ax.grid(True)
ax.set_ylim(70, 84)

plt.tight_layout()
# plt.show()
plt.savefig("./image/performance_comparison.png")
