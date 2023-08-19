import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
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
        print(name, T)
        if T < 1000:
            expl = np.concatenate([expl, expl[-1] * np.ones(1000 - T)])
        plt.plot(expl, color=colors[i], label=name)
    plt.legend(prop={'size': 17})
    plt.tick_params(labelsize=15)
    plt.ylim([-1, 20])
    plt.title("SP-based Algorithms (SAD)", fontsize=23)
    plt.xlabel("Step", fontsize=19)
    plt.ylabel("Exploitability", fontsize=19)
    plt.tight_layout()
    plt.savefig("figs/sad_sp_based.png", dpi=600)

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
        plt.plot(x, y, markers_lines[name], color=colors[i], label=labels[name])
    plt.legend(prop={'size': 17})
    plt.tick_params(labelsize=15)
    plt.ylim([-1, 20])
    plt.title("FXP and PSRO-based Algorithms (SAD)", fontsize=23)
    plt.xlabel("Step", fontsize=19)
    plt.ylabel("Exploitability", fontsize=19)
    plt.tight_layout()
    plt.savefig("figs/sad_psro_based.png", dpi=600)

    plt.cla()
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(8, 7))
    for i, name in enumerate(["FXP", "PSRO_NE"]):
        data = torch.load(f"models/{name}.pt")
        T = data["T"]
        end_point = data["end_point"]
        expl = np.stack(data["q"], 1)
        expl = np.sum(expl, 0)
        bg = 0
        title_name = name
        if title_name == "PSRO_NE":
            title_name = "PSRO-Nash"
        for ed in end_point[:10]:
            if bg > 0:
                axes[i].plot(np.arange(T)[bg:ed], expl[bg:ed], color=colors[i])
                axes[i].axvline(bg, linestyle=":", color=colors[i])
            else:
                axes[i].plot(np.arange(T)[bg:ed], expl[bg:ed], color=colors[i], label=title_name)
            bg = ed
        if i == 1:
            axes[i].set_xlabel("Step", fontsize=19)
        axes[i].set_ylabel("Exploitability", fontsize=19)
        axes[i].set_ylim([1, 18])
        axes[i].tick_params(labelsize=15)
        axes[i].legend(prop={'size': 17})
    plt.tight_layout()
    plt.savefig("figs/sad_fsp_psro.png", dpi=600)        


if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.makedirs("figs")
    
    print_sp()
    print_fxp_psro()
