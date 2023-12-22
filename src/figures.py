from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


palette = sns.color_palette("tab10")
saving_folder = os.path.abspath('results')

# Control separability for non-purified mAbs.
if True:
    # Create an array with the colors you want to use
    colors = [palette[3], palette[0], (0.4,0.4,0.4)]
    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))

    fig_name = 'Control separability for non-purified mAbs'
    data = pd.read_excel(os.path.join(saving_folder,fig_name+'.xlsx'))

    kwargs_box = {'data':data, 'x':'sample', 'y':'on_pert_max', 'color':'white'}
    # kwargs_strip = {'data':data, 'x':'sample', 'y':'on_pert_max', 'color':'black', 'size':10, 'alpha':.3}
    kwargs_strip = {'data':data, 'x':'sample', 'y':'on_pert_max', 'size':10, 'alpha':.9, 'linewidth':0.5, 'edgecolor':'gray', 'jitter':True}

    plt.figure(figsize=(12, 7))
    sns.boxplot(**kwargs_box)
    sns.stripplot(**kwargs_strip)
    plt.ylim(top=1.0)
    plt.xlabel("Tested controls", fontsize=15)
    plt.ylabel("Phagocytic score", fontsize=15)

    plt.text(0.30, 0.9, '****', transform=plt.gca().transAxes)
    plt.axhline(y=0.89, color='black', linewidth=.5, alpha=.7,xmin=0.15, xmax=0.5)
    plt.text(0.5, 0.95, '****', transform=plt.gca().transAxes)
    plt.axhline(y=0.94, color='black', linewidth=.5, alpha=.7,xmin=0.15, xmax=0.85)
    plt.text(0.67, 0.26, 'ns', transform=plt.gca().transAxes)
    plt.axhline(y=0.2, color='black', linewidth=.5, alpha=.7,xmin=0.5, xmax=0.85)

    plt.savefig(Path(saving_folder) / Path(fig_name + '.png'), dpi=300)
    print(f"# Saved {fig_name}")
    plt.close('all')


# Screening 96 mAbs using vOPA.
if True:
    fig_name = 'Screening 96 mAbs using vOPA'
    data = pd.read_excel(os.path.join(saving_folder,fig_name+'.xlsx'))

    colors = [palette[3], palette[0], (0.4,0.4,0.4)]
    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))

    # create a function to map scores to colors
    def score_to_color(score):
        if score <= 0.21:
            #gray
            return 'Low'
        if score <= 0.42:
            #green
            return 'Moderate'
        if score > 0.42:
            #red
            return 'High'

    def treatment_to_sample(x):
        if x == '2C7':
            return '2C7'
        elif x == 'NEGATIVE':
            return 'Unrelated mAb'
        else:
            return 'mAb'

    # create a new column for colors based on the score
    data["Phagocytosis-promoting level"] = data["on_pert_max"].apply(score_to_color)

    # create a new column for sample based on the treatment
    data["sample"] = data["treatment"].apply(treatment_to_sample)

    # sort the data dataframe by the treatment and on_pert_max column
    data['plot_order'] = data.loc[:,'sample'].map({'mAb':1, '2C7':2, 'Unrelated mAb':0})
    data = data.sort_values(by=['plot_order', 'on_pert_max'], ascending=[True, True]).reset_index(drop=True)

    # Trasform Phagocytosis-promoting level so that Unrelated mAb and 2C7 are named Controls
    data.loc[data['sample'] == 'Unrelated mAb', 'Phagocytosis-promoting level'] = 'Controls'
    data.loc[data['sample'] == '2C7', 'Phagocytosis-promoting level'] = 'Controls'

    # data["ii"] = data.index.values + 1
    data["ii"] = data.index.values
    palette = sns.color_palette("tab10")
    # kwargs_bar_plot = {'data':data, 'x':'ii', 'y':'on_pert_max', 'hue':'Phagocytosis-promoting level', 'palette':((0.4,0.4,0.4),palette[2],palette[3])}
    kwargs_bar_plot = {'data':data, 'x':'ii', 'y':'on_pert_max', 'hue':'Phagocytosis-promoting level', 'palette':(palette[7],palette[0],palette[2],palette[3])}


    plt.figure(figsize=(20, 10))
    ax = sns.barplot(**kwargs_bar_plot)

    # Position legend in top left corner
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), title='Phagocytosis-promoting level', ncol=1, frameon=True)
    plt.xlabel("Anti-Gonococcus mAbs", fontsize=15, labelpad=-40)
    plt.ylabel("Phagocytic score", fontsize=15)

    # xticks = list(range(1,97)) + [98] + [105]
    # ticks to appear on the graph
    # xticks = data["ii"].values[96:]
    xticks = data['ii'].values[:6].tolist() + data["ii"].values[102:].tolist()
    # xticks_labels = [' ']*96
    # xticks_labels += ['Unrelated mAb', '2C7']
    xticks_labels = [' ', ' ', 'Unrelated mAb', ' ', ' ', ' ']
    xticks_labels += [' ', ' ', '2C7', ' ', ' ', ' ']

    # plt.xticks(ticks=xticks, labels=xticks_labels, rotation=30)
    plt.xticks(ticks=xticks, labels=xticks_labels, rotation=0)

    plt.savefig(Path(saving_folder) / Path(fig_name + '.png'), dpi=300)
    print(f"# Saved {fig_name}")
    plt.close('all')

    print('DONE')


# Concentration quantification for the 96 anti-gonococcus mAbs.
if True:
    fig_name = 'Concentration quantification vs Phagocytic score'
    data = pd.read_excel(os.path.join(saving_folder,fig_name+'.xlsx'))

    # create a function to map scores to colors
    def score_to_color(score):
        if score <= 0.21:
            #grigio
            return 'Low'
        if score <= 0.42:
            #verde
            return 'Moderate'
        if score > 0.42:
            #rosso
            return 'High'

    # create a new column for colors based on the score
    data["Phagocytosis-promoting level"] = data["Phagocytic score"].apply(score_to_color)

    # modify the data Concentration values substituting 0 with 0.001
    data.loc[data['Concentration'] == 0, 'Concentration'] = 0.001

    my_palette = sns.color_palette("tab10")

    plt.figure(figsize=(12, 8))

    sns.scatterplot(data=data, x='Phagocytic score', y='Concentration', hue='Phagocytosis-promoting level', palette=((0.4,0.4,0.4),palette[2],palette[3]), s=100, alpha=0.9, edgecolor="gray")
    plt.axhline(y = 0.05, color = 'gray', linestyle = '--')

    plt.yscale('log')
    my_ticks = np.array((0.001, 0.05, 0.5, 5, 50))
    # plt.yticks(ticks=np.log(my_ticks+1e10), labels=(0, 0.05, 0.5, 5, 50))
    plt.yticks(ticks=my_ticks, labels=(0, 0.05, 0.5, 5, 50))
    plt.xlim(right=1.)
    plt.xlabel("Phagocytic score", fontsize=15)
    plt.ylabel("µg/ml", fontsize=15)

    plt.savefig(Path(saving_folder) / Path(fig_name + '.png'), dpi=300)
    print(f"# Saved {fig_name}")
    plt.close('all')
    