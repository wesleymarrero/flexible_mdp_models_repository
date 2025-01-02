# ==================================================
# Creating Policy and Action Value Function Plots
# ==================================================

# Loading modules
import os  # directory changes
import numpy as np  # matrix operations
import pandas as pd  # data frame operations
import matplotlib # plotting configurations
matplotlib.use("Agg") # making sure interactive mode is off (use "TkAgg" for interactive mode)
import matplotlib.pyplot as plt #base plots
plt.ioff() # making sure interactive mode is off (use plt.ion() for interactive mode)
import seaborn as sns #plots

# Plotting parameters
sns.set_style("ticks")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 6, 'font.size': 12}
sns.set_context("paper", rc=paper_rc) # use also for talks

## Function to plot total QALYs saved over planning horizon by risk group
def plot_qalys_saved(qalys_df):

    # Figure parameters
    mks = ['o', 'X', 'D', 's', 'P', '^']
    n_colors = 8  # number of colors in palette
    xlims = [0.5, 10.5] # limit of the x-axis
    x_ticks = range(2, 12, 2)#[1, 5, 10]
    y_ll = 0 # lower label in the y-axis
    y_ul = int(np.ceil(qalys_df['qalys'].max())) # upper label in the y-axis
    ylims = [y_ll-0.2, y_ul+0.2] # limits of the y-axis
    y_ticks = np.arange(y_ll, y_ul+0.5, 0.5)
    axes_size = 14 # font size of axes labels
    subtitle_size = 12 # font size for subplot titles
    tick_size = 10 # font size for tick labels
    legend_size = 10 # font size for legend labels
    cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']

    # Making figure
    fig, axes = plt.subplots(nrows=1, ncols=3)
    sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[qalys_df['bp_cat']==cat_labels[1]], style="policy", markers=mks,
                 dashes=False, palette=sns.color_palette("Greys", n_colors)[1:-1], errorbar=None, ax=axes[0]) #
    sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[qalys_df['bp_cat']==cat_labels[2]], style="policy", markers=mks,
                 dashes=False, palette=sns.color_palette("Greys", n_colors)[1:-1], errorbar=None, legend=False, ax=axes[1])
    sns.lineplot(x="year", y="qalys", hue="policy", data=qalys_df[qalys_df['bp_cat']==cat_labels[3]], style="policy", markers=mks,
                 dashes=False, palette=sns.color_palette("Greys", n_colors)[1:-1], errorbar=None, legend=False, ax=axes[2])

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Expected Life-Years Saved\n(in Millions)', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='',
             ylim=ylims, yticks=y_ticks, ylabel='')
    fig.subplots_adjust(bottom=0.2, wspace=0.3)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)

    axes[0].set_title('Elevated BP', fontsize=subtitle_size, fontweight='semibold')
    axes[1].set_title('Stage 1 Hypertension', fontsize=subtitle_size, fontweight='semibold')
    axes[2].set_title('Stage 2 Hypertension', fontsize=subtitle_size, fontweight='semibold')

    # Modifying Legend
    handles, labels = axes[0].get_legend_handles_labels()
    order = range(6)
    handles = [x for _, x in sorted(zip(order, handles))]
    labels = [x for _, x in sorted(zip(order, labels))]
    axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.7, -0.2),
                      handles=handles, labels=labels, prop={'size': legend_size})

    # Printing plot
    fig.set_size_inches(7.5, 3) # for simplified panel plots in paper
    if qalys_df.policy.unique()[0] in ['Optimal Policy', 'SBBI', 'On-Policy MC', 'Off-Policy MC',
                                       'Sarsa', 'Q-learning']:
        plt.savefig("QALYs Saved - Algorithm Comparison.pdf", bbox_inches='tight')
    elif qalys_df.policy.unique()[0] in ['Best in SBBI-SBMCC', 'Best in TD-SVP', 'Median in SBBI-SBMCC',
                                         'Median in TD-SVP', 'Fewest in SBBI-SBMCC', 'Fewest in TD-SVP']:
        plt.savefig("QALYs Saved - Algorithm Comparison SVP.pdf", bbox_inches='tight')
    plt.close()

## Function to plot distribution of treatment by risk group at year 1 and 10
def plot_trt_dist(trt_df):

    # Figure parameters
    n_colors = 6 # number of colors in palette
    xlims = [-0.5, 3.5] # limit of the x-axis
    ylims = [-0.5, 5.5] # limits of the y-axis
    y_ticks = np.arange(6)
    axes_size = 10 # font size of axes labels
    subtitle_size = 9 # font size for subplot titles
    tick_size = 8 # font size for tick labels
    legend_size = 8 # font size for legend labels
    cat_labels = trt_df.bp_cat.unique() # BP categories
    flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5, markersize=2,
                      linestyle='none') # outliers properties

    # Overall
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==1)],
                palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[0])
    sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==10)],
                palette=sns.color_palette("Greys", n_colors), linewidth=0.75, flierprops=flierprops, ax=axes[1])

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('BP Category', fontsize=axes_size, fontweight='semibold') #'Risk Group'
    plt.ylabel('Number of Medications', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xlabel='',
             ylim=ylims, yticks=y_ticks, ylabel='')
    fig.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.45)

    # Adding subtitles
    axes[0].set_title('Year 1', fontsize=subtitle_size, fontweight='semibold')
    axes[1].set_title('Year 10', fontsize=subtitle_size, fontweight='semibold')
    fig.subplots_adjust(bottom=0.3)  # for overal plots

    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=15, fontsize=tick_size-1, ticks=np.arange(len(cat_labels)), labels=cat_labels, ha='center') # for overall plots
        plt.yticks(fontsize=tick_size)

    # Modifying Legend
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.15, -0.35),
                      handles=handles, labels=labels, prop={'size': legend_size})
    axes[1].get_legend().remove()

    # Saving plot
    fig.set_size_inches(6.5, 3.5) # overall
    plt.savefig("Treatment per Policy - Algortihm Comparison SVP.pdf", bbox_inches='tight')

    plt.close()
