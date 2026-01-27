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

def analyze(folder, n_runs, baselines=False, log=False, normalize_regret=True, output_file=None, hide_baselines=False, start_no=0):
    avg_regret = None
    abl_regret = None
    UCRL_regret = None
    UCRL2_regret = None
    UCRL3_regret = None
    KL_regret = None

    avg_time = None
    abl_time = None
    UCRL_time = None
    UCRL2_time = None
    UCRL3_time = None
    KL_time = None

    avg_loss = None
    abl_loss = None
    UCRL_loss = None
    UCRL2_loss = None
    UCRL3_loss = None
    KL_loss = None

    for run_no in range(start_no, start_no+n_runs):
        with open(f"{folder}/run_{run_no}", "rb") as f:
            run_data = pickle.load(f)

        if baselines:
            baselines_file = f"{folder}/baselines_{run_no}"
            if os.path.isfile(baselines_file):
                with open(baselines_file, "rb") as f:
                    baseline_data = pickle.load(f)
            elif os.path.isfile(f"{folder}/baselines_run_{run_no}"):
                baselines_file = f"{folder}/baselines_run_{run_no}"
                with open(baselines_file, "rb") as f:
                    baseline_data = pickle.load(f)
            else:
                with open(f"{folder}/no_UCRL3_baselines_{run_no}", "rb") as f:
                    baseline_data = pickle.load(f)
                with open(f"{folder}/UCRL3_baselines_{run_no}", "rb") as f:
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
            #UCRL_regret = [[x] for x in normalize(run_data["UCRL"]["regret"])]
            avg_loss = [get_loss(run_data["rc"], run_data["ideal_gain"])]
            abl_loss = [get_loss(run_data["ablation"], run_data["ideal_gain"])]
            #UCRL_loss = [get_loss(run_data["UCRL"], run_data["ideal_gain"])]
            if baselines:
                UCRL2_regret = [[x] for x in normalize(baseline_data["UCRL2"]["regret"])]
                UCRL3_regret = [[x] for x in normalize(baseline_data["UCRL3"]["regret"])]
                KL_regret = [[x] for x in normalize(baseline_data["KL"]["regret"])]
                UCRL2_loss = [get_loss(baseline_data["UCRL2"], run_data["ideal_gain"])]
                UCRL3_loss = [get_loss(baseline_data["UCRL3"], run_data["ideal_gain"])]
                KL_loss = [get_loss(baseline_data["KL"], run_data["ideal_gain"])]
        else:
            for i, x in enumerate(normalize(run_data["rc"]["regret"])):
                avg_regret[i].append(x)
            for i, x in enumerate(normalize(run_data["ablation"]["regret"])):
                abl_regret[i].append(x)
            avg_loss.append(get_loss(run_data["rc"], run_data["ideal_gain"]))
            abl_loss.append(get_loss(run_data["ablation"], run_data["ideal_gain"]))
            if baselines:
                for i, x in enumerate(normalize(baseline_data["UCRL2"]["regret"])):
                    UCRL2_regret[i].append(x)
                for i, x in enumerate(normalize(baseline_data["UCRL3"]["regret"])):
                    UCRL3_regret[i].append(x)
                for i, x in enumerate(normalize(baseline_data["KL"]["regret"])):
                    KL_regret[i].append(x)
                UCRL2_loss.append(get_loss(baseline_data["UCRL2"], run_data["ideal_gain"]))
                UCRL3_loss.append(get_loss(baseline_data["UCRL3"], run_data["ideal_gain"]))
                KL_loss.append(get_loss(baseline_data["KL"], run_data["ideal_gain"]))

        #plt.plot(normalize(run_data["rc"]["regret"]), "b")
        #plt.plot(normalize(run_data["ablation"]["regret"]), "r")
        #plt.plot(normalize(run_data["UCRL"]["regret"]), "g")
        #plt.show()
    xlabels = [x*10000 for x in range(1000)]
    avg_regret_std = [np.std(x) for x in avg_regret]
    abl_regret_std = [np.std(x) for x in abl_regret]
    avg_regret = [np.mean(x) for x in avg_regret]
    abl_regret = [np.mean(x) for x in abl_regret]
    if baselines:
        UCRL3_regret_std = [np.std(x) for x in UCRL3_regret]
        UCRL2_regret_std = [np.std(x) for x in UCRL2_regret]
        KL_regret_std = [np.std(x) for x in KL_regret]
        UCRL3_regret = [np.mean(x) for x in UCRL3_regret]
        UCRL2_regret = [np.mean(x) for x in UCRL2_regret]
        KL_regret = [np.mean(x) for x in KL_regret]
    #plt.fill_between(xlabels,np.array(avg_regret)-np.array(avg_regret_std),(np.array(avg_regret)+np.array(avg_regret_std)), alpha=0.9, color="b")
    #plt.fill_between(xlabels,np.array(abl_regret)-np.array(abl_regret_std),(np.array(abl_regret)+np.array(abl_regret_std)), alpha=0.9, color="r")
    if baselines:
        linewidth = LINEWIDTH
        alpha = ALPHA
        if hide_baselines:
            linewidth /= 1.5
            alpha /= 3
        plt.plot(xlabels,KL_regret,"C2", label="KL_UCRL", linewidth=linewidth, alpha=alpha)
        #plt.fill_between(xlabels,np.array(KL_regret)-np.array(KL_regret_std),(np.array(KL_regret)+np.array(KL_regret_std)), alpha=0.1, color="g")
        plt.plot(xlabels,UCRL2_regret,"C3", label="UCRL2", linewidth=linewidth, alpha=alpha)
        #plt.fill_between(xlabels,np.array(UCRL2_regret)-np.array(UCRL2_regret_std),(np.array(UCRL2_regret)+np.array(UCRL2_regret_std)), alpha=0.1, color="orange")
        plt.plot(xlabels,UCRL3_regret,"C4", label="UCRL3", linewidth=linewidth, alpha=alpha)
        #plt.fill_between(xlabels,np.array(UCRL3_regret)-np.array(UCRL3_regret_std),(np.array(UCRL3_regret)+np.array(UCRL3_regret_std)), alpha=0.1, color="pink")
    if log:
        plt.yscale("log")
        plt.ylim(bottom=10000, top=10000000)
    else:
        plt.ylim(bottom=0, top=2500000)
        pass

    plt.plot(xlabels,avg_regret,"C0", label="UCRL-TSAC", linewidth=LINEWIDTH, alpha=ALPHA)
    line, = plt.plot(xlabels,abl_regret,"C1", label="Ablation", linewidth=LINEWIDTH, alpha=ALPHA)
    plt.plot(xlabels,avg_regret,"C0", label="UCRL-TSAC", linewidth=LINEWIDTH/2, alpha=ALPHA)

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
        print("Total regret (UCRL2): ", UCRL2_regret[-1])
    
    print("Reward ratio (rc): ", np.mean(avg_loss))
    print("Reward ratio (ablation): ", np.mean(abl_loss))
    if baselines:
        print("Reward ratio (UCRL2): ", np.mean(UCRL2_loss))
        print("Reward ratio (UCRL3): ", np.mean(UCRL3_loss))
        print("Reward ratio (KL): ", np.mean(KL_loss))


if __name__ == "__main__":
    if False:
        analyze("exp_out/path_11_states/", 50, True, False, True, "viz/path_11_states.pdf", True)
        analyze("exp_out/path_21_states/", 50, True, False, True, "viz/path_21_states.pdf", True)
        analyze("exp_out/path_51_states/", 50, True, False, True, "viz/path_51_states.pdf", True)
    elif False:
        analyze("exp_out/11_states/", 50, True, False, False, "viz/11_states.pdf", False)
        analyze("exp_out/21_states/", 50, True, False, False, "viz/21_states.pdf", False)
        analyze("exp_out/51_states/", 50, True, False, False, "viz/51_states.pdf", False)
    else:
        analyze("exp_out/11_states_2/", 50, True, False, False, "viz/11_states_2.pdf", False, start_no=0)
        analyze("exp_out/21_states_2/", 50, True, False, False, "viz/21_states_2.pdf", False, start_no=0)
        analyze("exp_out/51_states_2/", 50, True, False, False, "viz/51_states_2.pdf", False, start_no=0)
    lines = [
        plt.Line2D([0], [0], color="C0", lw=2, label="UCRL-RC"),
        plt.Line2D([0], [0], color="C1", lw=2, label="Ablation"),
        plt.Line2D([0], [0], color="C2", lw=2, label="KL-UCRL"),
        plt.Line2D([0], [0], color="C3", lw=2, label="UCRL2"),
        plt.Line2D([0], [0], color="C4", lw=2, label="UCRL3"),
    ]

    fig_legend = plt.figure(figsize=(5,1))
    fig_legend.legend(handles=lines, loc="center", ncol=5, frameon=False)
    fig_legend.savefig("viz/legend.pdf", bbox_inches="tight")

