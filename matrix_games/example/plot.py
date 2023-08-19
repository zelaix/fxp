import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch


def print_sp():
    names = ["SP", "FSP", "FoReL", "Replicator", "MWU", "CFR"]
    colors = [sns.color_palette()[i] for i in range(len(names))]

    plt.cla()
    plt.figure(figsize=(8, 6.5))
    for i, name in enumerate(names):
        data = torch.load(f"models/{name}.pt")
        T = data["T"]
        expl = np.stack(data["q"], 1)
        expl = np.sum(expl, 0)
        if T < 1000:
            expl = np.concatenate([expl, expl[-1] * np.ones(1000 - T)])
        print(name, T, f"expl={expl[-1]}")
        plt.plot(expl, color=colors[i], label=name)
    plt.legend(prop={'size': 17})
    plt.tick_params(labelsize=15)
    plt.ylim([-0.1, 5])
    plt.title("SP-based Algorithms (Example Game)", fontsize=21)
    plt.xlabel("Step", fontsize=19)
    plt.ylabel("Exploitability", fontsize=19)
    plt.tight_layout()
    plt.savefig("figs/example_sp_based.png", dpi=600)

def print_fxp_psro():
    names = ["FXP", "FXP_reset", "PSRO_NE", "PSRO_NE_warm", "PSRO_uniform", "PSRO_uniform_warm", "ODO", "ODO_warm"]
    labels = {
        "FXP": "FXP",
        "FXP_reset": "FXP (reset)",
        "PSRO_NE": "PSRO (Nash)",
        "PSRO_NE_warm": "PSRO (Nash) w.o. reset",
        "PSRO_uniform": "PSRO (uniform)",
        "PSRO_uniform_warm": "PSRO (uniform) w.o. reset",
        "ODO": "Online DO",
        "ODO_warm": "Online DO w.o. reset",
    }
    markers_lines = {
        "FXP": "*--",
        "FXP_reset": "*:",
        "PSRO_NE": "o--",
        "PSRO_NE_warm": "o:",
        "PSRO_uniform": "x--",
        "PSRO_uniform_warm": "x:",
        "ODO": "+--",
        "ODO_warm": "+:",
    }
    colors = [sns.color_palette()[i] for i in range(len(names))]

    plt.cla()
    plt.figure(figsize=(8, 6.5))
    for i, name in enumerate(names):
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
        plt.plot(x, y, markers_lines[name], color=colors[i], label=labels[name])
    plt.legend(prop={'size': 17})
    plt.tick_params(labelsize=15)
    plt.ylim([-0.1, 5])
    plt.title("FXP and PSRO-based Algorithms (Example Game)", fontsize=21)
    plt.xlabel("Step", fontsize=19)
    plt.ylabel("Exploitability", fontsize=19)
    plt.tight_layout()
    plt.savefig("figs/example_psro_based.png", dpi=600)


if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.makedirs("figs")
    
    print_sp()
    print_fxp_psro()