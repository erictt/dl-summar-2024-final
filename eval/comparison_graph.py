import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Adapter": ["MH", "Out", "Double", "Inv + Double"],
    "32": [78.48453790052511, 78.56006070686142, 80.16030422326828, None],
    "16": [80.0521476761068, 80.31803632386513, 81.22953410085692, None],
    "8": [81.52592408783839, 81.26012946094882, 82.20740610600804, None],
    "4": [81.91815525007279, 81.95067804161363, 82.79272097494145, 82.86460658011505],
}

df_grouped = pd.DataFrame(data)
fig, ax = plt.subplots(figsize=(14, 6))

bar_width = 0.2
adapters = df_grouped["Adapter"]
indices = range(len(adapters))

ax.bar(
    [i - 1.5 * bar_width for i in indices],
    df_grouped["32"],
    width=bar_width,
    label="32",
)
ax.bar(
    [i - 0.5 * bar_width for i in indices],
    df_grouped["16"],
    width=bar_width,
    label="16",
)
ax.bar(
    [i + 0.5 * bar_width for i in indices], df_grouped["8"], width=bar_width, label="8"
)
ax.bar(
    [i + 1.5 * bar_width for i in indices], df_grouped["4"], width=bar_width, label="4"
)

# baseline
ax.axhline(
    y=82.91251169582613, color="r", linestyle="--", label="deepset/roberta-base-squad2"
)

ax.set_xlabel("Adapter Method")
ax.set_ylabel("F1 Score")
ax.set_title("Adapter Training Results")
ax.legend(title="Reduction Factor", loc="lower left")
ax.set_xticks(indices)
ax.set_xticklabels(adapters)
ax.grid(True)
ax.set_ylim(70, 84)

plt.show()
