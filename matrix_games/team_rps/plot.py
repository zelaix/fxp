import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch


def print_exploitability():
    sp_names = ["SP", "FSP"]
    fxp_psro_names = ["PSRO_uniform", "PSRO_uniform_warm", "FXP"]
    colors = [sns.color_palette()[i] for i in range(len(sp_names + fxp_psro_names))]

    plt.cla()
    plt.figure(figsize=(8, 6.5))
    for i, name in enumerate(sp_names):
        data = torch.load(f"models/{name}.pt")
        T = data["T"]
        expl = np.stack(data["q"], 1)
        expl = np.sum(expl, 0)
        if T < 1000:
            expl = np.concatenate([expl, expl[-1] * np.ones(1000 - T)])
        print(name, T, f"expl={expl[-1]}")
        plt.plot(expl, color=colors[i], label=name)

    for i, name in enumerate(fxp_psro_names):
        print(name)
        data = torch.load(f"models/{name}_meta.pt")
        end_point = data["end_point"]
        expl = np.stack(data["q"], 1)
        expl = np.sum(expl, 0)
        x = np.array(end_point[:-1])
        x = x[:(x < 1200).sum()+1]
        if x[-1] > 1300:
            x[-1] = 1300
        x = np.concatenate([[0], x])
        print(x)
        y = expl[x]
        print("expl=", y[-1], "expl point=")
        for _x, _y in zip(x, y):
            print(f"({_x}, {np.round(_y, 2)})", end=" ")
        print()
        print()
        plt.plot(x, y, color=colors[i + 2], label=name)


    plt.legend(prop={'size': 17})
    plt.tick_params(labelsize=15)
    plt.ylim([-0.1, 1.1])
    plt.title("Team RPS", fontsize=21)
    plt.xlabel("Step", fontsize=19)
    plt.ylabel("Exploitability", fontsize=19)
    plt.tight_layout()
    plt.savefig("figs/team_rps_exploitability.png", dpi=600)


if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.makedirs("figs")
    
    print_exploitability()
