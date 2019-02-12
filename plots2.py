import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skopt.plots import plot_convergence
from skopt import gp_minimize, dump, load, forest_minimize, dummy_minimize
from skopt.plots import plot_evaluations, plot_objective
from scipy.optimize import OptimizeResult

from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator

def my_plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `true_minimum` [float, optional]:
        The true minimum value of the function, if known.

    * `yscale` [None or string, optional]:
        The scale for the y-axis.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            mins = [np.min(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")

    return ax

def my_plot_ls(*args, **kwargs):
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Physical Parameters Convergence")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min Cost(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, list):
            n_calls = len(results)
            mins = [np.min(results[:i])
                    for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)
    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")

    return ax



#lego
#train rake hook tap from left draw
#test rake push
#lego_rand = [0.20854831913754657, 0.21179450781221448, 0.23488503965113763, 0.2589614061399911, 0.2495135976358629, 0.22485777640991444, 0.2507502868762444, 0.2162233559049789, 0.23384707115378148, 0.22512401597788217, 0.2307547342317895, 0.252880160878473, 0.24043828237566567, 0.20495126158150656, 0.25108387513076136, 0.2160315170744414, 0.23715448574555786, 0.2707272613523407, 0.24559608104546599, 0.22522698241207334, 0.2497300698940435, 0.24172704931083464, 0.23867242357283677, 0.24110687533570446, 0.2576748548491826, 0.24213707933304768, 0.22743316734500296, 0.2469331188581493, 0.22630116479560994, 0.24204774231669404, 0.2392565952726381, 0.23512773952203253, 0.2376970938444735, 0.250537970652845, 0.22742485923987238, 0.2615837837028607, 0.2277534243062196, 0.20466552025919818, 0.22761650521312457, 0.2431298858412767, 0.24437223581793782, 0.23874732972931034, 0.23458601755523245, 0.23819394948581044, 0.2679776287656442, 0.24199552101576327, 0.25070514529037624, 0.25307128153449443, 0.22119312446379885, 0.23044515122770748]
lego_rand = [0.24199480816728566, 0.24386811312165999, 0.24857778369630382, 0.23838101849874532, 0.2427983709829834, 0.23200730431944547, 0.2445809311746812, 0.23640954698629763, 0.25191906917364415, 0.22064627395051511, 0.2405884011113838, 0.23403324556812655, 0.2522527089868172, 0.23460469290197816, 0.23597197583426327, 0.23240759651808435, 0.24722314430875345, 0.24145795127467795, 0.2281961068590579, 0.24286355134845306, 0.2348602150757754, 0.2502373717404914, 0.25036859182072485, 0.23517922343991485, 0.24237738910636103, 0.24516334328421663, 0.23637972866255408, 0.2445756697039907, 0.22722709099780491, 0.2507384441417877, 0.2345236865822188, 0.21247014095968875, 0.23701168204349055, 0.23264526905924252, 0.22919344192778057, 0.23782654168490874, 0.22425165501842154, 0.24656239025668386, 0.2767691837729388, 0.24285982997721484, 0.25629945924566544, 0.24163613131137554, 0.256747400876685, 0.23410312224288737, 0.24487252788412212, 0.246777770564075, 0.2239263580168247, 0.23225597967517964, 0.2545140183689756, 0.24129261636680208]


#lego_rand_best = [0.20854831913754657, 0.20495126158150656, 0.20466552025919818]
#lego_gen = [0.07517530465639993, 0.07251796356841474, 0.07030841299442803]

lego_res = [0.23720338639509875, 0.2327838674715662, 0.2338488692055507, 0.23093093443402646, 0.23678776033391152, 0.2458450614406149, 0.24060796322216654, 0.2514481511068168, 0.2385885498497517, 0.25391161013394425, 0.22858693939203475, 0.270399680551946, 0.24649229946353537, 0.22320768734991475, 0.22813761746352954, 0.22245085005488924, 0.23008541654315262, 0.2576206346252853, 0.23948501862175028, 0.2091863169158099, 0.2361100062108123, 0.23812867435900972, 0.23477916902811038, 0.22441842693160052, 0.23430903183336527, 0.2395198258371916, 0.22566661362196733, 0.23245760299091794, 0.236818157606079, 0.25017982541334444, 0.20827809540210043, 0.23342344018193806, 0.22010086887015168, 0.24296967070038644, 0.23483194803391644, 0.2517702306562959, 0.23912041955237545, 0.24362364451838134, 0.24307623410097864, 0.23027366610761626, 0.2611493745916591, 0.23586452699143806, 0.2518697523020348, 0.2458685842139333, 0.23627729024870583, 0.22703595259465503, 0.2569906720513816, 0.2181617774976199, 0.2358745576685699, 0.22845708646957777]
lego_best = [0.23720338639509875, 0.2327838674715662, 0.23093093443402646, 0.22858693939203475, 0.22320768734991475, 0.22245085005488924, 0.2091863169158099, 0.20827809540210043]
lego_gen = [0.01732400387238834, 0.021016331984946998, 0.016365857458242088, 0.012141062719089608, 0.02061264866612956, 0.017180708099616138, 0.017598826401634654, 0.01303158361533862]


best_ind = [lego_res.index(elem) for elem in lego_best]
#best_ind.append(len(ps))
lego_gen2 = [10.0 for elem  in lego_res]
for idx,bind in enumerate(best_ind):
    lego_gen2[bind]=0.165+4*lego_gen[idx]

fig = plt.Figure()
#dummy_res = load("random50.bz2") #load("dummy.bz2")
#gp_res = load("gp50.bz2")
#gp_res = load("lego_latf_weight50.bz2") #load("dummy.bz2")
#gp_res = load("rollf_weight.bz2")

#import copy
#res_gen = copy.deepcopy(gp_res)
#res_gen.func_vals[:]=1.0


#plot = my_plot_convergence(#("random", dummy_res),
#                        ("gp", gp_res),
#                        ("gp", res_gen),
#                         yscale="log")
#                        #true_minimum=0.1)

#plot = my_plot_ls(("GP",lego_res),
#                  ("New Action", lego_gen2),
#                  ("Random", lego_rand),
#                  yscale="log")

#plot.legend(loc="best", prop={'size': 6}, numpoints=1)

#fig.savefig("_sim.png")

gp_res = load("/home/vizzy/Dropbox/dsl_proj/pybullet_port/rollf_weight.bz2")
_ = plot_evaluations(gp_res)
_ = plot_objective(gp_res)

plt.show()

input()