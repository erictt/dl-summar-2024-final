import json

import matplotlib.pyplot as plt

adapters = ["seq_bn_default", "seq_bn_mh_default", "double_seq_bn_inv_default"]

for adapter in adapters:
    with open(f"./train/output/{adapter}/trainer_state.json", "r") as f:
        log_history = json.load(f)["log_history"]
    print(log_history)
    steps = [entry["step"] for entry in log_history[:-1]]
    losses = [entry["loss"] for entry in log_history[:-1]]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker="o", linestyle="-")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss over Steps")
    plt.grid(True)
    # plt.show()
    plt.savefig(f"./image/loss_{adapter}.png")
