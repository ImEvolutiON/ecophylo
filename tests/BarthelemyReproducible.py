import ecophylo
import numpy as np
from io import StringIO
import pandas as pd
import plotnine as p9
from pathlib import Path
from scipy.stats import linregress

# (i) an exploration of the influence of past demographic fluctuations on
# diversity patterns


n = 500
mu = 1e-4
nsim = 100
tau = 0


times = [1e2, 1e3, 1e4, 1e5]


for i in range(len(times)):
    ds = [[9e5, [99e4,8e7, "uniform"]]] # reduction
    changetimes = [[0,times[i]]]
    ecophylo.dosimuls(nsim = nsim,
                      samples = n,
                      deme_sizes= ds,
                      mu = mu,
                      tau = tau,
                      spmodel = "NTB",
                      changetimes= changetimes,
                      output=['Params','Sumstat'],
                      file_name = f"/home/theo/PhD/Ecophylo-Dev/output_{i}.csv")

results = []

for i in range(len(times)):
    file = Path(f"/home/theo/PhD/Ecophylo-Dev/output_{i}.csv")
    with open(file) as f:
        txt = f.read()
        
        params_txt, sumstat_txt = txt.split("###")
        
        params_df = pd.read_csv(
            StringIO(params_txt),
            sep=r"\s+"
            )
        sumstat_df = pd.read_csv(
            StringIO(sumstat_txt),
            sep=r"\s+")
        
        tmp = pd.DataFrame({
            "x": np.log(
                params_df["deme_sizes_pop0_t0"] /
                params_df["deme_sizes_pop0_t1"]
                ),
            "y": sumstat_df["alpha0"],
            "file": file.stem
            })
    results.append(tmp)

result = pd.concat(results, ignore_index=True)


nplot = result["file"].nunique()

plot = (
    p9.ggplot(result, p9.aes(x="x", y="y"))
    + p9.geom_point(alpha=0.5)
    + p9.facet_grid("file ~ .")
    + p9.theme_bw()
)

plot.save(
    "/home/theo/PhD/mosaicGENEALOGY.png",
    width=2,
    height=5*nplot,
    dpi=300,
    limitsize=False
)















# (i) an exploration of the influence of past demographic fluctuations on
# diversity patterns

import ecophylo
n = 500
mu = 0.001
nsim = 100


times = [1e2, 5e2, 1.5e3, 2e3, 3e3]


for i in range(len(times)):
    ds = [[1e4, [1e3,5e3, "uniform"]]] # reduction
    changetimes = [[0,times[i]]]
    ecophylo.dosimuls(nsim = nsim,
                      samples = n,
                      deme_sizes= ds,
                      mu = mu,
                      tau = 2,
                      spmodel = "paraphyletic",
                      age = "mutation",
                      changetimes= changetimes,
                      output=['Params','Sumstat'],
                      file_name = f"/home/theo/PhD/Ecophylo-Dev/output_{i}.csv")

results = []
r2 = list()

for i in range(len(times)):
    file = Path(f"/home/theo/PhD/Ecophylo-Dev/output_{i}.csv")
    with open(file) as f:
        txt = f.read()
        
        params_txt, sumstat_txt = txt.split("###")
        
        params_df = pd.read_csv(
            StringIO(params_txt),
            sep=r"\s+"
            )
        sumstat_df = pd.read_csv(
            StringIO(sumstat_txt),
            sep=r"\s+")
        
        tmp = pd.DataFrame({
            "x": params_df["deme_sizes_pop0_t0"] /
                params_df["deme_sizes_pop0_t1"],
            "y": sumstat_df["alpha0"],
            "file": file.stem
            })
        reg = linregress(tmp["x"], tmp["y"])
        r2tmp = reg.rvalue**2
        r2.append(r2tmp)
        
    results.append(tmp)

result = pd.concat(results, ignore_index=True)


nplot = result["file"].nunique()

plot = (
    p9.ggplot(result, p9.aes(x="x", y="y"))
    + p9.geom_point(alpha=0.5)
    + p9.facet_grid("file ~ .")
    + p9.geom_smooth(method="lm", se = True, color = "blue")
    + p9.theme_bw()
)

plot.save(
    "/home/theo/PhD/ExpansionSGD.png",
    width=2,
    height=5*nplot,
    dpi=300,
    limitsize=False
)