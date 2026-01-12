from experiment import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as pe
import pickle
import os

LINEWIDTH = 4
FONTSIZE = 18
TICKSIZE = 14
ALPHA = 1.0

def analyze(folder, n_runs, baselines=False, log=False, normalize_regret=True, output_file=None):
    avg_regret = None
    abl_regret = None
    ucrl_regret = None
    ucrl2_regret = None
    ucrl3_regret = None
    kl_regret = None

    avg_time = None
    abl_time = None
    ucrl_time = None
    ucrl2_time = None
    ucrl3_time = None
    kl_time = None

    avg_loss = None
    abl_loss = None
    ucrl_loss = None
    ucrl2_loss = None
    ucrl3_loss = None
    kl_loss = None

    for run_no in range(n_runs):
        with open(f"{folder}/run_{run_no}", "rb") as f:
            run_data = pickle.load(f)
        if baselines:
            baselines_file = f"{folder}/baselines_{run_no}"
            if os.path.isfile(baselines_file):
                with open(baselines_file, "rb") as f:
                    baseline_data = pickle.load(f)
            else:
                with open(f"{folder}/no_ucrl3_baselines_{run_no}", "rb") as f:
                    baseline_data = pickle.load(f)
                with open(f"{folder}/ucrl3_baselines_{run_no}", "rb") as f:
                    baseline_data |= pickle.load(f)

        #print(f"total reward: {run_data['rc']['reward'][-1]}")
        #print(f"total time: {run_data['rc']['time'][-1]}")
        #print(f"avg gain: {run_data['rc']['reward'][-1]/run_data['rc']['time'][-1]}")
        #print(f"ideal gain: {run_data['rc']['ideal_gain']}")
        def normalize(l):
            if normalize_regret:
                return [x/run_data["ideal_gain"] for x in l]
            return [x for x in l]
        def get_loss(rd, ideal_gain):
            return rd["reward"][-1]/(ideal_gain*rd["time"][-1])
        if avg_regret is None:
            avg_regret = [[x] for x in normalize(run_data["rc"]["regret"])]
            abl_regret = [[x] for x in normalize(run_data["ablation"]["regret"])]
            #ucrl_regret = [[x] for x in normalize(run_data["ucrl"]["regret"])]
            avg_loss = [get_loss(run_data["rc"], run_data["ideal_gain"])]
            abl_loss = [get_loss(run_data["ablation"], run_data["ideal_gain"])]
            #ucrl_loss = [get_loss(run_data["ucrl"], run_data["ideal_gain"])]
            if baselines:
                ucrl2_regret = [[x] for x in normalize(baseline_data["ucrl2"]["regret"])]
                ucrl3_regret = [[x] for x in normalize(baseline_data["ucrl3"]["regret"])]
                kl_regret = [[x] for x in normalize(baseline_data["kl"]["regret"])]
                ucrl2_loss = [get_loss(baseline_data["ucrl2"])]
                ucrl3_loss = [get_loss(baseline_data["ucrl3"])]
                kl_loss = [get_loss(baseline_data["kl"])]
        else:
            for i, x in enumerate(normalize(run_data["rc"]["regret"])):
                avg_regret[i].append(x)
            for i, x in enumerate(normalize(run_data["ablation"]["regret"])):
                abl_regret[i].append(x)
            avg_loss.append(get_loss(run_data["rc"], run_data["ideal_gain"]))
            abl_loss.append(get_loss(run_data["ablation"], run_data["ideal_gain"]))
            if baselines:
                for i, x in enumerate(normalize(baseline_data["ucrl2"]["regret"])):
                    ucrl2_regret[i].append(x)
                for i, x in enumerate(normalize(baseline_data["ucrl3"]["regret"])):
                    ucrl3_regret[i].append(x)
                for i, x in enumerate(normalize(baseline_data["kl"]["regret"])):
                    kl_regret[i].append(x)
                ucrl2_loss.append(get_loss(baseline_data["ucrl2"]))
                ucrl3_loss.append(get_loss(baseline_data["ucrl3"]))
                kl_loss.append(get_loss(baseline_data["kl"]))

        #plt.plot(normalize(run_data["rc"]["regret"]), "b")
        #plt.plot(normalize(run_data["ablation"]["regret"]), "r")
        #plt.plot(normalize(run_data["ucrl"]["regret"]), "g")
        #plt.show()
    xlabels = [x*10000 for x in range(1000)]
    avg_regret_std = [np.std(x) for x in avg_regret]
    abl_regret_std = [np.std(x) for x in abl_regret]
    avg_regret = [np.mean(x) for x in avg_regret]
    abl_regret = [np.mean(x) for x in abl_regret]
    if baselines:
        ucrl3_regret_std = [np.std(x) for x in ucrl3_regret]
        ucrl2_regret_std = [np.std(x) for x in ucrl2_regret]
        kl_regret_std = [np.std(x) for x in kl_regret]
        ucrl3_regret = [np.mean(x) for x in ucrl3_regret]
        ucrl2_regret = [np.mean(x) for x in ucrl2_regret]
        kl_regret = [np.mean(x) for x in kl_regret]
    plt.plot(xlabels,avg_regret,"C0", label="UCRL-TSAC", linewidth=LINEWIDTH, alpha=ALPHA)
    line, = plt.plot(xlabels,abl_regret,"C1", label="Ablation", linewidth=LINEWIDTH, alpha=ALPHA)
    plt.plot(xlabels,avg_regret,"C0", label="UCRL-TSAC", linewidth=LINEWIDTH/2, alpha=ALPHA)
    #plt.fill_between(xlabels,np.array(avg_regret)-np.array(avg_regret_std),(np.array(avg_regret)+np.array(avg_regret_std)), alpha=0.1, color="b")
    #plt.fill_between(xlabels,np.array(abl_regret)-np.array(abl_regret_std),(np.array(abl_regret)+np.array(abl_regret_std)), alpha=0.1, color="r")
    if baselines:
        plt.plot(xlabels,kl_regret,"C2", label="KL_UCRL", linewidth=LINEWIDTH, alpha=ALPHA)
        #plt.fill_between(xlabels,np.array(kl_regret)-np.array(kl_regret_std),(np.array(kl_regret)+np.array(kl_regret_std)), alpha=0.1, color="g")
        plt.plot(xlabels,ucrl2_regret,"C3", label="UCRL2", linewidth=LINEWIDTH, alpha=ALPHA)
        #plt.fill_between(xlabels,np.array(ucrl2_regret)-np.array(ucrl2_regret_std),(np.array(ucrl2_regret)+np.array(ucrl2_regret_std)), alpha=0.1, color="orange")
        plt.plot(xlabels,ucrl3_regret,"C4", label="UCRL3", linewidth=LINEWIDTH, alpha=ALPHA)
        #plt.fill_between(xlabels,np.array(ucrl3_regret)-np.array(ucrl3_regret_std),(np.array(ucrl3_regret)+np.array(ucrl3_regret_std)), alpha=0.1, color="pink")
    if log:
        plt.yscale("log")
        plt.ylim(bottom=100, top=1700000)
    else:
        plt.ylim(bottom=0, top=1700000)
        pass
    plt.xlabel("Steps", fontsize=FONTSIZE)
    plt.ylabel("Total Regret " + (" (Normalized)" if normalize_regret else ""), fontsize=FONTSIZE)
    plt.xlim(left=0, right=10000000)
    #plt.legend()
    plt.grid(True)
    plt.tick_params(axis="both", labelsize=TICKSIZE)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(TICKSIZE)
    ax.xaxis.get_offset_text().set_fontsize(TICKSIZE)

    if output_file is not None:
        plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.show()

    #plt.plot(gain, "b")
    #plt.show()
    
    #print("Total reward: ", )
    print("Total regret: ", avg_regret[-1])
    print("Total regret (ablation): ", abl_regret[-1])
    if baselines:
        print("Total regret (ucrl2): ", ucrl2_regret[-1])
    
    print("Reward ratio (rc): ", np.mean(avg_loss))
    print("Reward ratio (ablation): ", np.mean(abl_loss))
    if baselines:
        print("Reward ratio (ucrl2): ", np.mean(ucrl2_loss))
        print("Reward ratio (ucrl3): ", np.mean(ucrl3_loss))
        print("Reward ratio (kl): ", np.mean(kl_loss))


if __name__ == "__main__":
    analyze("exp_out/11_states/", 50, False, False, True, "viz/11_states.pdf")
    analyze("exp_out/21_states/", 50, False, False, True, "viz/21_states.pdf")
    analyze("exp_out/51_states/", 49, False, False, True, "viz/51_states.pdf")
    lines = [
        plt.Line2D([0], [0], color="C0", lw=2, label="UCRL-TSAC"),
        plt.Line2D([0], [0], color="C1", lw=2, label="Ablation"),
        #plt.Line2D([0], [0], color="C2", lw=2, label="UCRL2"),
        #plt.Line2D([0], [0], color="C3", lw=2, label="KL-UCRL"),
        #plt.Line2D([0], [0], color="C4", lw=2, label="UCRL3"),
    ]

    fig_legend = plt.figure(figsize=(5,1))
    fig_legend.legend(handles=lines, loc="center", ncol=5, frameon=False)
    fig_legend.savefig("viz/legend.pdf", bbox_inches="tight")

