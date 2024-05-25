import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.optimize import curve_fit


matplotlib.rc('font', **{'size': 8})

val_pps = np.array([32.23533, 11.7897])

accs = np.array([
    [0.5025,0.499,0.5035,0.4955,0.5005,0.4995,0.4725,0.3095,0.4955,0.533,0.524,0.2775,0.4085,0.346,0.738,0.41,0.4635,0.2375,0.4745,0.3865,0.0,0.3435,0.0],
    [0.542,0.566,0.497,0.498,0.5115,0.4875,0.5275,0.916,0.959,0.569,0.747,0.97,0.272,0.4445,1.0,0.485,0.182,0.4845,0.8875,0.7545,0.5255,0.7215,0.672],
    [0.7315,0.798,0.5585,0.5455,0.691,0.7445,0.466,0.648,0.921,0.566,0.54,0.9725,0.619,0.547,0.66,0.5615,0.3655,0.5805,0.933,0.3765,0.4325,0.7335,0.774]
    ])

surprisals = np.array([
    [0.03918489999999996,0.052712049999999996,0.04509825000000005,0.036567750000000024,0.04171755000000001,0.040404099999999964,0.02908199999999998,0.060496150000000144,0.048931199999999855,0.06360724999999995,0.0356695,0.03177190000000003,0.04928149999999994,0.031954599999999986,0.024783500000000017,0.025090350000000015,0.026956150000000036,0.06073510000000005,0.047093050000000025,0.035760400000000074,0.03475054999999992,0.030447299999999948,0.057594450000000005],
    [0.19240054999999973,0.20048084999999974,0.26571174999999997,0.24195075000000033,0.17617720000000012,0.2773000999999997,0.2600731000000002,0.5913790499999995,0.40320309999999965,0.3858653999999996,0.17320399999999977,0.36182574999999984,0.15877575000000008,0.24985040000000003,0.4926903999999994,0.17245365000000004,0.316237650000001,0.18696389999999988,0.314774300000001,0.19950960000000018,0.05298390000000002,0.3505760500000001,0.3332372500000002],
    [0.3561102000000006,0.40605469999999994,0.28876360000000034,0.3724788500000003,0.3339502499999994,0.5385458500000004,0.14299384999999978,0.3691256000000006,0.6494529500000003,0.6924968999999995,0.21940390000000032,0.7334909500000008,0.19602850000000005,0.25754869999999935,0.30983340000000026,0.3362695500000006,0.2975408000000004,0.3750502500000002,0.6881645499999981,0.35397060000000047,0.12206229999999985,0.556727399999999,0.2061480000000003],
])

headers = ['agreem. det. noun\nacross 1 adj.','agreem. det. noun\nbetween neighbors','agreem. subject verb\nacross prep. phrase','agreem. subject verb\nacross relative clause','agreem. subject verb\nin question with aux','agreem. subject verb\nin simple question','anaphor agreement\npronoun gender','arg. struct.\ndropped arg.','arg. struct.\nswapped args','arg. struct.\ntransitive','binding\nprinciple a','case subjective\npronoun','ellipsis\nn bar','filler-gap wh\nquestion object','filler-gap wh\nquestion subject','irregular verb','island effects\nadjunct island','island effects coord.\nstruct. constraint','local attractor in\nquestion with aux','npi licensing\nmatrix question','npi licensing only\nnpi licensor','quantifiers\nexistential there','quantifiers\nsuperlative','Average']


def perplexity():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(np.arange(2), val_pps, color='lightgray', edgecolor='black', marker='*', alpha=1, s=100, zorder=4)
    coefficients = np.polyfit(np.arange(2), val_pps, 1)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(min(np.arange(2)), max(np.arange(2)), 100)
    y_fit = polynomial(x_fit)
    ax.plot(x_fit, y_fit, color='dodgerblue', lw=4)

    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 15, 30], [0, 15, 30], fontsize=12)
    ax.set_xticks([0, 1], [1, 2], fontsize=12)

    ax.set_title('Validation set perplexity', fontsize=12)
    ax.set_xlabel('Age', fontsize=12)
    ax.tick_params(axis='both', which='both', length=10, width=3)

    plt.tight_layout()
    plt.savefig('pps.png', bbox_inches='tight')


def surprisal():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(np.arange(3), surprisals.mean(1), color='lightgray', edgecolor='black', marker='P', alpha=1, s=100, zorder=4)
    coefficients = np.polyfit(np.arange(3), surprisals.mean(1), 2)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(min(np.arange(3)), max(np.arange(3)), 100)
    y_fit = polynomial(x_fit)
    ax.plot(x_fit, y_fit, color='seagreen', lw=4)

    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(-0.1, 0.5)
    ax.set_yticks([0, 0.5], [0, 0.5], fontsize=12)
    ax.set_xticks([0, 1, 2], [0, 1, 2], fontsize=12)

    ax.set_title('Average difference in\nsurprisal between pairs', fontsize=12)
    ax.set_xlabel('Age', fontsize=12)
    ax.tick_params(axis='both', which='both', length=10, width=3)

    plt.tight_layout()
    plt.savefig('surprisal.png', bbox_inches='tight')


def accuracy():
    def fourier_series(x, *a):
        ret = a[0]
        n = (len(a) - 1) // 2
        for i in range(1, n + 1):
            ret += a[2 * i - 1] * np.cos(2 * np.pi * i * x) + a[2 * i] * np.sin(2 * np.pi * i * x)
        return ret

    n_harmonics = 1

    initial_guess = np.zeros(2 * n_harmonics + 1)


    fig, axs = plt.subplots(5, 5, figsize=(15, 12))

    for i in range(5):
        for j in range(5):
            cell = i * 5 + j
            if cell < 24:
                axs[i][j].axhline(y=0.5, color='silver', linestyle='-', linewidth=3, zorder=1)

                if cell < 23:
                    for a in range(3):
                        axs[i][j].scatter(a, accs[a][cell], color='lightgray', edgecolor='black', marker='P', alpha=1, s=50, zorder=4)

                    # params, params_covariance = curve_fit(fourier_series, np.arange(3), accs[:, cell], p0=initial_guess)
                    # y_fit = fourier_series(np.arange(3), *params)
                    coefficients = np.polyfit(np.arange(3), accs[:, cell], 2)
                    polynomial = np.poly1d(coefficients)
                    x_fit = np.linspace(min(np.arange(3)), max(np.arange(3)), 100)
                    y_fit = polynomial(x_fit)
                    axs[i][j].plot(x_fit, y_fit, color='black', lw=3)

                if cell == 23:
                    for a in range(3):
                        axs[i][j].scatter(a, accs[a].mean(), color='lightgray', marker='P', edgecolor='black', alpha=1, s=50, zorder=4)

                    # params, params_covariance = curve_fit(fourier_series, np.arange(3), accs.mean(axis=1), p0=initial_guess)
                    # y_fit = fourier_series(np.arange(3), *params)
                    coefficients = np.polyfit(np.arange(3), accs.mean(axis=1), 2)
                    polynomial = np.poly1d(coefficients)
                    x_fit = np.linspace(min(np.arange(3)), max(np.arange(3)), 100)
                    y_fit = polynomial(x_fit)
                    axs[i][j].plot(x_fit, y_fit, color='black', lw=3)
                
                axs[i][j].spines['left'].set_linewidth(3)
                axs[i][j].spines['bottom'].set_linewidth(3)

                axs[i][j].spines['top'].set_visible(False)
                axs[i][j].spines['right'].set_visible(False)
                axs[i][j].set_ylim(-0.1, 1.1)
                axs[i][j].set_xlim(-0.5, 3.0)
                axs[i][j].set_yticks([0.5, 1], [0.5, 1], fontsize=15)
                axs[i][j].set_xticks([0, 1, 2], [0, 1, 2], fontsize=15)
                axs[i][j].tick_params(axis='both', which='both', length=10, width=3)

                axs[i][j].set_title(headers[cell], fontsize=15)
            
            if cell in range(0, 24, 5):
                axs[i][j].set_ylabel('Accuracy', fontsize=15)
            
            if cell in [20, 21, 22, 23]:
                axs[i][j].set_xlabel('Age', fontsize=15)

    plt.tight_layout()
    plt.savefig('plot.png', bbox_inches='tight')


if __name__ == '__main__':
    perplexity()
    surprisal()