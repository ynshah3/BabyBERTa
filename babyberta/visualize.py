import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.optimize import curve_fit


matplotlib.rc('font', **{'size': 8})

accs = np.array([
    [0.5025,0.499,0.5035,0.4955,0.5005,0.4995,0.4725,0.3095,0.4955,0.533,0.524,0.2775,0.4085,0.346,0.738,0.41,0.4635,0.2375,0.4745,0.3865,0.0,0.3435,0.0],
    [0.542,0.566,0.497,0.498,0.5115,0.4875,0.5275,0.916,0.959,0.569,0.747,0.97,0.272,0.4445,1.0,0.485,0.182,0.4845,0.8875,0.7545,0.5255,0.7215,0.672],
    [0.7315,0.798,0.5585,0.5455,0.691,0.7445,0.466,0.648,0.921,0.566,0.54,0.9725,0.619,0.547,0.66,0.5615,0.3655,0.5805,0.933,0.3765,0.4325,0.7335,0.774],
    [0.875,0.856,0.6215,0.5935,0.8225,0.862,0.5025,0.464,0.9315,0.622,0.711,0.9315,0.516,0.9565,0.8925,0.663,0.613,0.982,0.936,0.508,0.5235,0.8235,0.666],
    [0.8835,0.8865,0.635,0.606,0.845,0.882,0.5385,0.722,0.9365,0.642,0.7585,0.906,0.5205,0.8985,0.8735,0.6845,0.616,0.939,0.954,0.646,0.7085,0.8195,0.6715],
    [0.8545,0.847,0.6565,0.666,0.8815,0.8715,0.5135,0.82,0.854,0.6665,0.764,0.7895,0.465,0.938,0.713,0.6855,0.568,0.941,0.907,0.901,0.718,0.802,0.5135]
    ])

surprisals = np.array([
    [0.03918489999999996,0.052712049999999996,0.04509825000000005,0.036567750000000024,0.04171755000000001,0.040404099999999964,0.02908199999999998,0.060496150000000144,0.048931199999999855,0.06360724999999995,0.0356695,0.03177190000000003,0.04928149999999994,0.031954599999999986,0.024783500000000017,0.025090350000000015,0.026956150000000036,0.06073510000000005,0.047093050000000025,0.035760400000000074,0.03475054999999992,0.030447299999999948,0.057594450000000005],
    [0.19240054999999973,0.20048084999999974,0.26571174999999997,0.24195075000000033,0.17617720000000012,0.2773000999999997,0.2600731000000002,0.5913790499999995,0.40320309999999965,0.3858653999999996,0.17320399999999977,0.36182574999999984,0.15877575000000008,0.24985040000000003,0.4926903999999994,0.17245365000000004,0.316237650000001,0.18696389999999988,0.314774300000001,0.19950960000000018,0.05298390000000002,0.3505760500000001,0.3332372500000002],
    [0.3561102000000006,0.40605469999999994,0.28876360000000034,0.3724788500000003,0.3339502499999994,0.5385458500000004,0.14299384999999978,0.3691256000000006,0.6494529500000003,0.6924968999999995,0.21940390000000032,0.7334909500000008,0.19602850000000005,0.25754869999999935,0.30983340000000026,0.3362695500000006,0.2975408000000004,0.3750502500000002,0.6881645499999981,0.35397060000000047,0.12206229999999985,0.556727399999999,0.2061480000000003],
    [4.1022798249999886,3.8842777999999987,3.6481881999999968,3.675127724999996,3.676281274999995,3.645388125000002,3.9238702249999946,3.8960163500000005,4.083802124999994,4.7814199249999945,4.265394000000008,4.0120159,4.347483775000002,4.083213950000001,4.249039999999998,3.621501050000002,3.9938016000000056,3.884762,3.9749302749999997,4.016663975000003,4.162927074999993,3.6713957500000056,3.7938875249999957],
    [4.188414224999998,4.027877350000004,3.668857650000002,3.7898949999999987,3.7580832750000037,3.755599624999998,4.017635400000001,3.9779805500000034,4.172813699999997,4.831631149999999,4.335378699999995,4.085950924999995,4.503547274999999,4.09289737500001,4.252165550000008,3.8031728500000024,4.1301695000000045,3.866120050000005,3.995823675000006,4.101856199999995,4.278333174999989,3.8483071999999927,3.7928390000000016],
    [4.2250322749999984,4.017579024999999,3.761479574999997,3.8074221249999995,3.9048368750000044,3.77236277499999,3.958645649999998,4.017485375000006,4.2184292249999995,4.803730999999992,4.288119799999996,4.155987499999994,4.585493024999992,4.277142700000002,4.313844975000001,3.830875225000013,4.166176925000005,3.9513011499999937,3.9071493000000084,4.226057024999998,4.346224525000001,3.8711944249999943,3.7204833999999996]
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
    def fourier_series(x, *a):
        ret = a[0]
        n = (len(a) - 1) // 2
        for i in range(1, n + 1):
            ret += a[2 * i - 1] * np.cos(2 * np.pi * i * x) + a[2 * i] * np.sin(2 * np.pi * i * x)
        return ret

    n_harmonics = 2

    initial_guess = np.zeros(2 * n_harmonics + 1)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(np.arange(6), surprisals.mean(1), color='lightgray', edgecolor='black', marker='P', alpha=1, s=100, zorder=4)
    coefficients = np.polyfit(np.arange(3), surprisals.mean(1)[:3], 1)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(-0.5, 2.5, 100)
    y_fit = polynomial(x_fit)
    ax.plot(x_fit, y_fit, color='seagreen', lw=4)
    coefficients = np.polyfit([3, 4, 5], surprisals.mean(1)[3:], 1)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(2.5, 5.5, 100)
    y_fit = polynomial(x_fit)
    ax.plot(x_fit, y_fit, color='seagreen', lw=4)

    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.5, 6.0)
    ax.set_ylim(-0.1, 5.0)
    ax.set_yticks([0, 5], [0, 5], fontsize=12)
    ax.set_xticks([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], fontsize=12)

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

    n_harmonics = 2

    initial_guess = np.zeros(2 * n_harmonics + 1)


    fig, axs = plt.subplots(6, 4, figsize=(16, 24))

    for i in range(6):
        for j in range(4):
            cell = i * 4 + j
            if cell < 24:
                axs[i][j].axhline(y=0.5, color='silver', linestyle='-', linewidth=3, zorder=1)

                if cell < 23:
                    for a in range(6):
                        axs[i][j].scatter(a, accs[a][cell], color='lightgray', edgecolor='black', marker='P', alpha=1, s=50, zorder=4)

                    # params, params_covariance = curve_fit(fourier_series, np.arange(6), accs[:, cell], p0=initial_guess)
                    # y_fit = fourier_series(np.arange(6), *params)
                    coefficients = np.polyfit(np.arange(6), accs[:, cell], 3)
                    polynomial = np.poly1d(coefficients)
                    x_fit = np.linspace(min(np.arange(6)), max(np.arange(6)), 100)
                    y_fit = polynomial(x_fit)
                    axs[i][j].plot(x_fit, y_fit, color='black', lw=3)

                if cell == 23:
                    for a in range(6):
                        axs[i][j].scatter(a, accs[a].mean(), color='lightgray', marker='P', edgecolor='black', alpha=1, s=50, zorder=4)

                    # params, params_covariance = curve_fit(fourier_series, np.arange(6), accs.mean(axis=1), p0=initial_guess)
                    # y_fit = fourier_series(np.arange(6), *params)
                    coefficients = np.polyfit(np.arange(6), accs.mean(axis=1), 2)
                    polynomial = np.poly1d(coefficients)
                    x_fit = np.linspace(min(np.arange(6)), max(np.arange(6)), 100)
                    y_fit = polynomial(x_fit)
                    axs[i][j].plot(x_fit, y_fit, color='black', lw=3)
                
                axs[i][j].spines['left'].set_linewidth(3)
                axs[i][j].spines['bottom'].set_linewidth(3)

                axs[i][j].spines['top'].set_visible(False)
                axs[i][j].spines['right'].set_visible(False)
                axs[i][j].set_ylim(-0.1, 1.1)
                axs[i][j].set_xlim(-0.5, 6.0)
                axs[i][j].set_yticks([0.5, 1], [0.5, 1], fontsize=15)
                axs[i][j].set_xticks([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], fontsize=15)
                axs[i][j].tick_params(axis='both', which='both', length=10, width=3)

                axs[i][j].set_title(headers[cell], fontsize=15)
            
            if cell in range(0, 24, 4):
                axs[i][j].set_ylabel('Accuracy', fontsize=15)
            
            if cell in [20, 21, 22, 23]:
                axs[i][j].set_xlabel('Age', fontsize=15)

    plt.tight_layout()
    plt.savefig('plot.png', bbox_inches='tight')


if __name__ == '__main__':
    # accuracy()
    surprisal()