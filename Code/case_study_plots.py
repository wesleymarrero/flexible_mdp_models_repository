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
sns.set_context("paper", rc=paper_rc)

# Main paper functions
## Function to plot convergence of confidence interval width with different number of batches and a fix number of observations per batch
def plf_cov_batch(ci_width, plot_batches, selected_batches):

    # Figure parameters
    axes_size = 14 # font size of axes labels
    tick_size = 12 # font size for tick labels

    # Making figure
    plt.figure()
    plt.plot(ci_width[:plot_batches], color='black')

    # Modifying Axes
    plt.xlabel('Number of Batches', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Confidence Interval Width', fontsize=axes_size, fontweight='semibold')
    plt.xticks([1, selected_batches, plot_batches], fontsize=tick_size)
    plt.yticks(ticks=[int(np.floor(min(ci_width)*1000))/1000, int(np.floor(max(ci_width)*10))/10], fontsize=tick_size)
    plt.ylim(0, int(np.floor(max(ci_width)*10))/10+0.05)

    # Adding horizontal line for reference number of batches
    plt.hlines(y=ci_width[-1], xmin=1, xmax=plot_batches, color='red', alpha=0.60, zorder=100)
    plt.text(x=selected_batches, y=ci_width[selected_batches]+0.01, s=str(ci_width[selected_batches].round(2)), color='gray', fontsize=tick_size)
    plt.vlines(x=selected_batches, ymin=-0.05, ymax=ci_width[selected_batches], color='gray', linestyle='--', zorder=200)

    # Saving plot
    plt.savefig('Number of Batches Analysis.pdf', bbox_inches='tight')
    plt.close()

## Function to plot demographic summary by BP category
def plot_demo_bp(demo):

    # Figure parameters
    n_colors = 6 # number of colors in palette
    xlims = [-0.5, 3.5] # limit of the x-axis
    ylims = [0, 4] # limits of the y-axis
    y_ticks = np.arange(6)
    data_labe_size = 8 # font size for data labels
    axes_size = 10 # font size of axes labels
    subtitle_size = 9 # font size for subplot titles
    tick_size = 8 # font size for tick labels
    legend_size = 8 # font size for legend labels
    cat_labels = demo.bp_cat.unique()  # BP categories

    # Making plot
    fig, axes = plt.subplots(nrows=1, ncols=2)
    g1 = sns.barplot(x="bp_cat", y="wt", hue="sex", data=demo[demo.race == 0], palette=sns.color_palette("Greys", n_colors)[3:-1], ax=axes[0])
    g2 = sns.barplot(x="bp_cat", y="wt", hue="sex", data=demo[demo.race == 1], palette=sns.color_palette("Greys", n_colors)[3:-1], ax=axes[1])

    # Adding data labels
    for p in g1.patches:
        g1.annotate(format(p.get_height(), '.2f'), (p.get_x()+p.get_width()/2., p.get_height()), ha='center',
                       va='center', xytext=(0, 10), textcoords='offset points', fontsize=data_labe_size)

    for p in g2.patches:
        g2.annotate(format(p.get_height(), '.2f'), (p.get_x()+p.get_width()/2., p.get_height()), ha='center',
                       va='center', xytext=(0, 10), textcoords='offset points', fontsize=data_labe_size)

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('BP Catgory', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Number of People (in Millions)', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xlabel='',
             ylim=ylims, yticks=y_ticks, ylabel='')
    fig.subplots_adjust(bottom=0.12)

    ## Adding subtitles
    axes[0].set_title('Race: Black', fontsize=subtitle_size, fontweight='semibold')
    axes[1].set_title('Race: White', fontsize=subtitle_size, fontweight='semibold')

    for ax in fig.axes:
        plt.sca(ax)
        # plt.xticks(rotation=15, fontsize=tick_size-1, ticks=np.arange(len(risk_labels)), labels=risk_labels, ha='center')
        plt.xticks(fontsize=tick_size-1, ticks=np.arange(len(cat_labels)), labels=cat_labels,
                   ha='center')  # rotation=15,
        plt.yticks(fontsize=tick_size)

    # Modifying Legend
    handles, _ = axes[0].get_legend_handles_labels()
    labels = ['Females', 'Males']
    axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.1, -0.12),
                      handles=handles, labels=labels, prop={'size': legend_size})
    axes[1].get_legend().remove()

    fig.set_size_inches(6.5, 3.66)
    plt.savefig("Demographics by BP Category.pdf", bbox_inches='tight')
    plt.close()

## Function to plot total QALYs saved over planning horizon by risk group
def plot_qalys_saved(qalys_df):

    # Figure parameters
    mks = ['D', 's', 'o', 'P', 'X']
    n_colors = 7  # number of colors in palette
    xlims = [0.5, 10.5] # limit of the x-axis
    x_ticks = range(2, 12, 2)#[1, 5, 10]
    y_ll = 0 # lower label in the y-axis
    y_ul = int(np.ceil(qalys_df['qalys'].max()*10))/10 # upper label in the y-axis
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
    ## Panel plot
    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 2, 4, 1, 3]
    handles = [x for _, x in sorted(zip(order, handles))]
    labels = [x for _, x in sorted(zip(order, labels))]
    axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.7, -0.2),
                      handles=handles, labels=labels, prop={'size': legend_size})

    # Printing plot
    fig.set_size_inches(7.5, 3) # for simplified panel plots in paper
    plt.savefig("QALYs Saved per Policy.pdf", bbox_inches='tight')
    plt.close()

## Function to plot distribution of treatment by risk group at year 1 and 10
def plot_trt_dist(trt_df):

    # Figure parameters
    n_colors = 5 # number of colors in palette
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

    ## Adding subtitles
    bp_labels = ['Normal BP', 'Elevated BP', 'Stage 1\nHypertension', 'Stage 2\nHypertension']
    axes[0].set_title('Year 1', fontsize=subtitle_size, fontweight='semibold')
    axes[1].set_title('Year 10', fontsize=subtitle_size, fontweight='semibold')
    fig.subplots_adjust(bottom=0.3)  # for overal plots

    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=15, fontsize=tick_size-1, ticks=np.arange(len(bp_labels)), labels=bp_labels, ha='center') # for overall plots
        plt.yticks(fontsize=tick_size)

    # Modifying Legend
    ## For overall plot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.15, -0.35),
                      handles=handles, labels=labels, prop={'size': legend_size})
    axes[1].get_legend().remove()

    fig.set_size_inches(6.5, 3.5) # overall
    plt.savefig("Treatment per Policy.pdf", bbox_inches='tight')
    plt.close()

## Function to plot proportion of actions by policy convered in range by risk group
def plot_prop_covered(prop_df):

    # Figure parameters
    n_colors = 4  # number of colors in palette
    xlims = [0.5, 10.5] # limit of the x-axis
    x_ticks = range(2, 12, 2) #[1, 5, 10]
    ylims = [-0.1, 1.1] # limits of the y-axis
    y_ticks = [0, 0.5, 1]
    axes_size = 10 # font size of axes labels
    subtitle_size = 9 # font size for subplot titles
    tick_size = 8 # font size for tick labels
    legend_size = 8 # font size for legend labels
    line_width = 1.5 # width for lines in plots

    # Making figure
    fig, axes = plt.subplots(nrows=2, ncols=2)
    sns.lineplot(x="year", y="prop_cv", hue="bp_cat",
                 data=prop_df[(prop_df['race'] == 0) & (prop_df['sex'] == 0)],
                 style="bp_cat", markers=False, dashes=True, palette=sns.color_palette("Greys", n_colors)[1:], errorbar=None,
                 ax=axes[0, 0])
    sns.lineplot(x="year", y="prop_cv", hue="bp_cat",
                 data=prop_df[(prop_df['race'] == 0) & (prop_df['sex'] == 1)],
                 style="bp_cat", markers=False, dashes=True, palette=sns.color_palette("Greys", n_colors)[1:], errorbar=None,
                 legend=False, ax=axes[0, 1])
    sns.lineplot(x="year", y="prop_cv", hue="bp_cat",
                 data=prop_df[(prop_df['race'] == 1) & (prop_df['sex'] == 0)],
                 style="bp_cat", markers=False, dashes=True, palette=sns.color_palette("Greys", n_colors)[1:], errorbar=None,
                 legend=False, ax=axes[1, 0])
    sns.lineplot(x="year", y="prop_cv", hue="bp_cat",
                 data=prop_df[(prop_df['race'] == 1) & (prop_df['sex'] == 1)],
                 style="bp_cat", markers=False, dashes=True, palette=sns.color_palette("Greys", n_colors)[1:], errorbar=None,
                 legend=False, ax=axes[1, 1])

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Proportion of Patient-Years Covered in Range', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration
    plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='',
             ylim=ylims, yticks=y_ticks, ylabel='')
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.setp(ax.lines, linewidth=line_width)

    ## Adding subtitles
    axes[0, 0].set_title('Black Females', fontsize=subtitle_size, fontweight='semibold')
    axes[0, 1].set_title('Black Males', fontsize=subtitle_size, fontweight='semibold')
    axes[1, 0].set_title('White Females', fontsize=subtitle_size, fontweight='semibold')
    axes[1, 1].set_title('White Males', fontsize=subtitle_size, fontweight='semibold')

    ## Formatting y-axis as percentages
    for ax in fig.axes:
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])

    # Modifying Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.15, -1.9),
                      handles=handles[1:], labels=labels[1:], prop={'size': legend_size})

    fig.set_size_inches(6.5, 4.25)
    plt.savefig("Proportion of Patient-Years Covered in Range.pdf", bbox_inches='tight')
    plt.close()

## Function to plot range average (min-max) width and number of medications
def plot_ranges_len_meds(sens_df_width, sens_df_meds):

    # Figure parameters
    n_colors = 6  # number of colors in palette
    x_ticks = range(2, 12, 2)
    y_ul = int(np.ceil(sens_df_width['max'].max()/10))*10  # upper label in the y-axis
    ylims = [[0, y_ul+1], [-0.5, 5.5]] # limits of the y-axis
    y_ticks = [np.linspace(start=0, stop=y_ul, num=6), np.arange(6)]
    y_labels = ['Range Width', 'Number of Medications']
    axes_size = 10 # font size of axes labels
    subtitle_size = 9 # font size for subplot titles
    tick_size = 8 # font size for tick labels
    legend_size = 8 # font size for legend labels

    # Making figure
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.scatterplot(x="year", y="mean", hue="scenario", data=sens_df_width, markers=['H', 'v', '^', 'X'],
                    style="scenario", palette=np.sort(sns.color_palette("Greys", n_colors)[1:-1])[::-1].tolist(), linewidth=0.5,
                    ax=axes[0], zorder=100) # s=25,
    sns.scatterplot(x="year", y="mean", hue="scenario", data=sens_df_meds, markers=['H', 'v', '^', 'X'],
                    style="scenario", palette=np.sort(sns.color_palette("Greys", n_colors)[1:-1])[::-1].tolist(), linewidth=0.5,
                    legend=False, ax=axes[1], zorder=100) #s=25,

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    fig.subplots_adjust(bottom=0.2, wspace=0.3)
    for k, ax in list(enumerate(fig.axes))[:-1]:
        plt.sca(ax)
        plt.xlabel('')
        plt.xticks(ticks=x_ticks, fontsize=tick_size)
        plt.ylim(ylims[k])
        plt.yticks(ticks=y_ticks[k], fontsize=tick_size)
        plt.ylabel(y_labels[k], fontsize=subtitle_size, fontweight='semibold')

    # Adding error band (or bars)
    colors = sns.color_palette("Greys", n_colors).as_hex()[1:]
    for j, k, l in [(0, 0, 10), (1, 9, 20)]: # excluding random and worst future actions because they are the same as in best future action # enumerate(sens_df_width.scenario.unique())
        ## Range width plot
        axes[0].plot(sens_df_width.loc[sens_df_width.scenario==k, 'year'],
                     sens_df_width.loc[sens_df_width.scenario==k, 'min'],
                     marker='', linestyle='dashed', color=colors[j], zorder=l)  # for error bars: marker='_', markersize=mk_size, linestyle=''
        axes[0].plot(sens_df_width.loc[sens_df_width.scenario==k, 'year'],
                     sens_df_width.loc[sens_df_width.scenario==k, 'max'],
                     marker='', linestyle='dashed', color=colors[j], zorder=l) # for error bars: marker='_', markersize=mk_size, linestyle=''

        # Number of medications plot
        axes[1].plot(sens_df_meds.loc[sens_df_meds.scenario==k, 'year'],
                     sens_df_meds.loc[sens_df_meds.scenario==k, 'min'],
                     marker='', linestyle='dashed', color=colors[j], zorder=l)  # for error bars: marker='_', markersize=mk_size, linestyle=''
        axes[1].plot(sens_df_meds.loc[sens_df_meds.scenario==k, 'year'],
                     sens_df_meds.loc[sens_df_meds.scenario==k, 'max'],
                     marker='', linestyle='dashed', color=colors[j], zorder=l)  # for error bars: marker='_', markersize=mk_size, linestyle=''

    # Modifying Legend
    labels = ['Base Case', 'Assuming Normality', "Median in Next Year's Range", "Fewest in Next Year's Range"]
    handles, _ = axes[0].get_legend_handles_labels() # reversing order in handles (to follow the order of labels)
    axes[0].legend(loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(1.15, -0.2),
                   handles=handles[1:][::-1], labels=labels, prop={'size': legend_size})

    fig.set_size_inches(6.5, 3)
    plt.savefig("Range Width and Medications.pdf", bbox_inches='tight')
    plt.close()

## Function to plot proportion of actions by policy convered in range by misestimation scenario
def plot_prop_mis(prop_df):

    # Figure parameters
    mks = ['D', 'o']
    n_colors = 3  # number of colors in palette
    xlims = [0.5, 10.5] # limit of the x-axis
    x_ticks = range(2, 12, 2) #[1, 5, 10]
    ylims = [-0.1, 1.1] # limits of the y-axis
    y_ticks = [0, 0.5, 1]
    axes_size = 10 # font size of axes labels
    subtitle_size = 9 # font size for subplot titles
    tick_size = 8 # font size for tick labels
    legend_size = 8 # font size for legend labels
    line_width = 1.5 # width for lines in plots

    # Making figure
    fig, axes = plt.subplots(nrows=2, ncols=2)
    sns.lineplot(x="year", y="prop", hue="policy", data=prop_df[prop_df['misestimation']=='Base Case'], style="policy", markers=mks, #markeredgewidth=0.5,
                 dashes=False, errorbar=None, palette=sns.color_palette("Greys", n_colors)[1:], ax=axes[0, 0]) #sns.color_palette("Greys", n_colors)[1:] # "viridis"
    sns.lineplot(x="year", y="prop", hue="policy", data=prop_df[prop_df['misestimation']=='Half Event Rates'], style="policy", markers=mks, #markeredgewidth=0.5,
                 dashes=False, errorbar=None, palette=sns.color_palette("Greys", n_colors)[1:], legend=False, ax=axes[0, 1])
    sns.lineplot(x="year", y="prop", hue="policy", data=prop_df[prop_df['misestimation']=='Double Event Rates'], style="policy", markers=mks, #markeredgewidth=0.5,
                 dashes=False, errorbar=None, palette=sns.color_palette("Greys", n_colors)[1:], legend=False, ax=axes[1, 0])
    sns.lineplot(x="year", y="prop", hue="policy", data=prop_df[prop_df['misestimation']=='Half Treatment Benefit'], style="policy", markers=mks, #markeredgewidth=0.5,
                 dashes=False, errorbar=None, palette=sns.color_palette("Greys", n_colors)[1:], legend=False, ax=axes[1, 1])

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Proportion of Patients Covered in Range', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration
    plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='',
             ylim=ylims, yticks=y_ticks, ylabel='')
    fig.subplots_adjust(wspace=0.3, hspace=0.5, bottom=0.2)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        # plt.setp(ax.lines, linewidth=line_width)

    ## Adding subtitles
    axes[0, 0].set_title('Base Case', fontsize=subtitle_size, fontweight='semibold')
    axes[0, 1].set_title('Half Event Rates', fontsize=subtitle_size, fontweight='semibold')
    axes[1, 0].set_title('Double Event Rates', fontsize=subtitle_size, fontweight='semibold')
    axes[1, 1].set_title('Half Treatment Benefit', fontsize=subtitle_size, fontweight='semibold')

    ## Formatting y-axis as percentages
    for ax in fig.axes:
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])

    # Modifying Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.15, -1.8),
                      handles=handles[1:], labels=labels[1:], prop={'size': legend_size})

    # Saving plot
    fig.set_size_inches(6.5, 4.5)
    plt.savefig("Proportion of Patient-Years Covered in Range with Misestimated Parameters.pdf", bbox_inches='tight')
    plt.close()

## Function to plot ranges of near-optimal actions
def plot_range_actions(meds_df, medication_range):

    # Figure parameters
    n_colors = 6  # number of colors in palette
    x_ticks = range(2, 12, 2)
    xlims = [0.5, 10.5] # limit of the x-axis
    axes_size = 10 # font size of axes labels
    subtitle_size = 9 # font size for subplot titles
    tick_size = 8 # font size for tick labels
    legend_size = 8 # font size for legend labels
    mks = ['D', 'o', 'X'] # markers for plots
    marker_size = 24 # marker size

    # Making figure
    fig, axes = plt.subplots(nrows=2, ncols=2)
    sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==0], markers=mks,
                    style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
                    ax=axes[0, 0], zorder=100)
    sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==1], markers=mks,
                    style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
                    legend=False, ax=axes[0, 1], zorder=100)
    sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==2], markers=mks,
                    style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
                    legend=False, ax=axes[1, 0], zorder=100)
    sns.scatterplot(x="year", y="meds", hue="policy", data=meds_df[meds_df.id==3], markers=mks,
                    style="policy", s=marker_size, linewidth=0.5, palette=sns.color_palette("Greys", n_colors)[2:-1],
                    legend=False, ax=axes[1, 1], zorder=100)

    ## Adding subtitles
    axes[0, 0].set_title('54-year-old White Male', fontsize=subtitle_size, fontweight='semibold')
    axes[0, 1].set_title('54-year-old White Female', fontsize=subtitle_size, fontweight='semibold')
    axes[1, 0].set_title('54-year-old White Male Smoker', fontsize=subtitle_size, fontweight='semibold')
    axes[1, 1].set_title('70-year-old White Male', fontsize=subtitle_size, fontweight='semibold')

    # Figure Configuration
    ## Overall labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Year', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Antihypertensive Medications\n\n\n', fontsize=axes_size, fontweight='semibold')

    ## Values for y-axes
    all_meds = np.delete(np.round(np.arange(0, 5, 0.333333333), 2), 1)
    all_labels = ['0 SD/0 HD', '0 SD/1 HD', '1 SD/0 HD',
                  '0 SD/2 HD', '1 SD/1 HD', '2 SD/0 HD', # '2 SD/0 HD' = '0 SD/3 HD'
                  '1 SD/2 HD', '2 SD/1 HD', '3 SD/0 HD', # '2 SD/1 HD' = '0 SD/4 HD'  # '3 SD/0 HD' = '1 SD/3 HD'
                  '2 SD/2 HD', '3 SD/1 HD', '4 SD/0 HD', # '2 SD/2 HD' = '0 SD/5 HD'  # '3 SD/1 HD' = '1 SD/4 HD' # '4 SD/0 HD' = '2 SD/3 HD'
                  '3 SD/2 HD', '4 SD/1 HD', '5 SD/0 HD']

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xticks=x_ticks, xlabel='', ylabel='')
    fig.subplots_adjust(bottom=0.1, wspace=0.4, hspace=0.4)
    obs_meds = np.concatenate([meds_df.meds.to_numpy(), np.concatenate([x.min().to_numpy() for x in medication_range]),
                               np.concatenate([x.max().to_numpy() for x in medication_range])]) # same y axis across all plots (comment for individual axes and uncomment line inside loop)
    for k, ax in list(enumerate(fig.axes))[:-1]:
        plt.sca(ax)

        # shaded area
        plt.fill_between(x=np.arange(1, 11), y1=np.amin(medication_range[k], axis=0), y2=np.amax(medication_range[k], axis=0),
                         color=sns.color_palette("Greys", n_colors)[0], zorder=0)

        # x-axis
        plt.xticks(fontsize=tick_size)

        # y-axis
        # obs_meds = np.concatenate([meds_df[meds_df.id==k].meds.to_numpy(), medication_range[k].min().to_numpy(), medication_range[k].max().to_numpy()]) # individual axes (comment for smae y-axis across all plots)
        yticks = np.round(np.arange(np.amin(obs_meds), np.amax(obs_meds), 0.333333333), 2)
        yticks = yticks[yticks != 0.33] # Making sure that there is no tick between no treatment and a medication at half dosage
        if yticks.shape[0] == 0:
            yticks = [np.amin(obs_meds)]
        ind = np.where([j in list(yticks) for j in list(all_meds)])[0]
        ylabels = [all_labels[j] for j in ind.astype(int)]
        ax.set_yticks(ticks=yticks)
        ax.set_yticklabels(labels=ylabels, fontsize=tick_size)
        ax.set_ylim((np.amin(yticks)-0.15, np.amax(yticks)+0.15))

    # Modifying Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.1, -1.8),
                      handles=handles[1:], labels=labels[1:], prop={'size': legend_size}) # for 6 profiles: bbox_to_anchor=(1.1, -3.2) # for 4 profiles: bbox_to_anchor=(1.1, -1.8)

    #Saving plot
    fig.set_size_inches(6.5, 4) # for 6 profiles: fig.set_size_inches(6.5, 6) for 4 profiles: fig.set_size_inches(6.5, 4)
    plt.savefig('Ranges for Patient Profiles.pdf', bbox_inches='tight')
    plt.close()