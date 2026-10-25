from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from display_names import DATASET_DISPLAY_NAMES, METHOD_DISPLAY_NAMES
ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "tables_tex"
REAL_SOURCE = "original"
DISPLAY = METHOD_DISPLAY_NAMES
METHOD_GROUPS = [["gumbel_pred_xgb","gumbel_pred_ridge","xgboost_no_dataset_features","xgboost_monotone_llr","xgboost_per_window"],["isotonic","mlr","random_forest","svr","mlp","constant_threshold"],["gumbel99","mc99"],["gumbel999"]]
DATASET_ORDER = ["mesothelioma","pneumoconioses","als"]
def esc(value):
    text=str(value)
    for a,b in (("&",r"\&"),("%",r"\%"),("_",r"\_"),("#",r"\#")): text=text.replace(a,b)
    return text
def fmt(value,digits=4):
    return "--" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"
def write_table(filename,headers,rows,align,header_lines=None):
    lines=[r"\begin{tabular}{"+align+"}",r"\toprule"]
    lines.extend(header_lines if header_lines is not None else [" & ".join(headers)+r" \\"])
    lines.append(r"\midrule")
    previous=None
    for row in rows:
        if row.get("rule_before"):
            lines.append(row["rule_before"])
        elif previous is not None and row.get("group") is not None and row.get("group") != previous:
            lines.append(r"\midrule")
        if "raw_line" in row:
            lines.append(row["raw_line"])
            previous=row.get("group",previous)
            continue
        cells=[]
        for col in row["columns"]:
            value=row["display"][col]
            if col in row.get("bold",set()): value=r"\textbf{"+value+"}"
            cells.append(value)
        lines.append(" & ".join(cells)+r" \\")
        previous=row.get("group",previous)
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/filename).write_text("\n".join(lines)+"\n",encoding="utf-8")
def method_order(methods):
    return [m for group in METHOD_GROUPS for m in group if m in methods]
def group_for(method):
    return next(i for i,g in enumerate(METHOD_GROUPS) if method in g)
def bootstrap_ci(sums,counts):
    sums=np.asarray(sums,dtype=float); counts=np.asarray(counts,dtype=float); rng=np.random.default_rng(42)
    means=np.empty(1000,dtype=float)
    for i in range(len(means)):
        sample=rng.integers(0,len(sums),size=len(sums)); denominator=counts[sample].sum()
        means[i]=sums[sample].sum()/denominator if denominator else np.nan
    return np.quantile(means,[.025,.975])
def synthetic_table():
    w=pd.read_csv(TABLES/"window_metrics.csv")
    w=w[(w.scope=="synthetic_test")&(w.window_filter=="llr_positive")&(w.sample_stratum=="all")].copy()
    w["method"]=w.model.replace({"xgboost":"xgboost_per_window"})
    b=w.groupby("method").mae_p_0_01_to_0_10.mean(); f=w.groupby("method").f1.mean()
    d=pd.read_csv(TABLES/"dataset_metrics.csv"); d=d[d.scope=="synthetic_test"].copy(); d["method"]=d.model.replace({"xgboost":"xgboost_per_window"})
    c=d.groupby("method").absolute_log10_p_error.mean()
    predictions=pd.read_csv(ROOT/"results"/"predictions"/"synthetic_test_predictions.csv.gz")
    all_ids=np.sort(predictions.id.unique())
    predictions=predictions[(predictions.llr>0)&predictions.p_value.between(0.01,0.10)].copy()
    pred_columns={"gumbel_pred_xgb":"pred_p_gumbel_pred_xgb","gumbel_pred_ridge":"pred_p_gumbel_pred_ridge","xgboost_per_window":"pred_p_xgboost","xgboost_no_dataset_features":"pred_p_xgboost_no_dataset_features","xgboost_monotone_llr":"pred_p_xgboost_monotone_llr","isotonic":"pred_p_isotonic","mlr":"pred_p_mlr","random_forest":"pred_p_random_forest","svr":"pred_p_svr","mlp":"pred_p_mlp","gumbel99":"pred_p_gumbel99","mc99":"pred_p_mc99","gumbel999":"pred_p_gumbel999"}
    by_d={}
    for method,column in pred_columns.items():
        if column in predictions:
            errors=(predictions[column]-predictions.p_value).abs()
            grouped=errors.groupby(predictions.id).agg(["sum","count"])
            by_d[method]=(grouped["sum"].reindex(all_ids,fill_value=0).to_numpy(dtype=float),grouped["count"].reindex(all_ids,fill_value=0).to_numpy(dtype=float))
    methods=method_order(set(b.index)&set(f.index)&set(c.index)); eligible=[m for m in methods if (99 if m in ("mc99","gumbel99") else 999 if m=="gumbel999" else 0)<=99]
    rounded={m:{"band":float(f"{b[m]:.4f}"),"f1":float(f"{f[m]:.4f}" if f[m]>=0.9995 else f"{f[m]:.3f}"),"cluster":float(f"{c[m]:.4f}"),"replicates":99 if m in ("mc99","gumbel99") else 0} for m in eligible}
    best={"band":min(v["band"] for v in rounded.values()),"f1":max(v["f1"] for v in rounded.values()),"cluster":min(v["cluster"] for v in rounded.values()),"replicates":min(v["replicates"] for v in rounded.values())}; rows=[]
    for m in methods:
        reps=99 if m in ("gumbel99","mc99") else 999 if m=="gumbel999" else 0
        ci=bootstrap_ci(*by_d[m]) if m in by_d and len(all_ids)==90 else (np.nan,np.nan)
        raw={"band":float(b[m]),"f1":float(f[m]),"cluster":float(c[m]),"replicates":reps}; bold=set()
        if reps<=99:
            displayed={"band":float(f"{raw['band']:.4f}"),"f1":float(f"{raw['f1']:.4f}" if raw['f1'] >= 0.9995 else f"{raw['f1']:.3f}"),"cluster":float(f"{raw['cluster']:.4f}"),"replicates":reps}
            bold={key for key in displayed if key != "replicates" and displayed[key]==best[key]}
        f1_digits = 4 if raw["f1"] >= 0.9995 else 3
        rows.append({"group":group_for(m),"columns":["method","band","f1","cluster","replicates"],"display":{"method":esc(DISPLAY.get(m,m)),"band":f"{fmt(raw['band'])} [{fmt(ci[0])}, {fmt(ci[1])}]","f1":fmt(raw['f1'],f1_digits),"cluster":fmt(raw['cluster']),"replicates":str(reps)},"bold":bold})
    write_table("synthetic_test_comparison.tex",["Method",r"Band MAE [95\% bootstrap CI]","F1 @ 0.05",r"Cluster $|\Delta\log_{10}p|$","MC replicates"],rows,"lrrrr")
def paired_bootstrap():
    predictions=pd.read_csv(ROOT/"results"/"predictions"/"synthetic_test_predictions.csv.gz")
    all_ids=np.sort(predictions.id.unique())
    prediction_columns={"gumbel_pred_xgb":"pred_p_gumbel_pred_xgb","gumbel_pred_ridge":"pred_p_gumbel_pred_ridge","xgboost_per_window":"pred_p_xgboost","xgboost_no_dataset_features":"pred_p_xgboost_no_dataset_features","xgboost_monotone_llr":"pred_p_xgboost_monotone_llr","isotonic":"pred_p_isotonic","mlr":"pred_p_mlr","random_forest":"pred_p_random_forest","svr":"pred_p_svr","mlp":"pred_p_mlp","gumbel99":"pred_p_gumbel99","mc99":"pred_p_mc99","gumbel999":"pred_p_gumbel999"}
    band_rows=predictions[(predictions.llr>0)&predictions.p_value.between(0.01,0.10)]
    band_data={}
    for method,column in prediction_columns.items():
        if column in band_rows:
            err=(band_rows[column]-band_rows.p_value).abs()
            band_data[method]=err.groupby(band_rows.id).mean().reindex(all_ids)
    band_wide=pd.DataFrame(band_data,index=all_ids)
    metrics=pd.read_csv(TABLES/"dataset_metrics.csv")
    metrics=metrics[metrics.scope=="synthetic_test"].copy()
    metrics["id"]=pd.to_numeric(metrics["id"])
    metrics["method"]=metrics.model.replace({"xgboost":"xgboost_per_window"})
    cluster_wide=metrics.pivot_table(index="id",columns="method",values="absolute_log10_p_error",aggfunc="mean").reindex(all_ids)
    methods=method_order(set(band_wide.columns)&set(cluster_wide.columns))
    valid_band=band_wide[methods].notna().all(axis=1).to_numpy()
    rng=np.random.default_rng(42)
    bootstrap_indices=rng.integers(0,len(all_ids),size=(1000,len(all_ids)))
    records=[]
    for metric,values,valid in (("band_mae",band_wide,valid_band),("cluster_error",cluster_wide,np.ones(len(all_ids),dtype=bool))):
        baseline=values["gumbel_pred_xgb"].to_numpy(dtype=float)
        for method in methods:
            if method=="gumbel_pred_xgb":
                continue
            differences=values[method].to_numpy(dtype=float)-baseline
            paired=differences[valid]
            samples=np.empty(len(bootstrap_indices),dtype=float)
            for repeat,indices in enumerate(bootstrap_indices):
                selected=indices[valid[indices]]
                samples[repeat]=np.mean(differences[selected])
            low,high=np.quantile(samples,[0.025,0.975])
            records.append({"metric":metric,"method":method,"display_name":DISPLAY.get(method,method),"n_test_datasets":len(all_ids),"n_paired_datasets":int(valid.sum()),"mean_difference":float(np.mean(paired)),"ci95_low":float(low),"ci95_high":float(high),"surrogate_better_fraction":float(np.mean(samples>0))})
    result=pd.DataFrame(records)
    output=TABLES/"paired_bootstrap.csv"
    result.to_csv(output,index=False)
    print(result.to_csv(index=False),end="")
    return result
def source_real():
    if REAL_SOURCE=="original":
        real=pd.read_csv(TABLES/"real_all_windows.csv"); repeats=pd.read_csv(TABLES/"baseline_repeats.csv")
        return real,repeats[(repeats.row_type=="summary")&repeats.statistic.isin(["mean","std","min","max"])]
    path=TABLES/"real_vs_reference.csv"
    if not path.exists(): raise FileNotFoundError(f"Reference metrics are unavailable: {path}")
    frame=pd.read_csv(path); needed={"dataset","method","statistic","hard_f1","hard_mae_log10_p"}; missing=needed-set(frame.columns)
    if missing: raise ValueError(f"Reference metrics missing columns: {sorted(missing)}")
    return frame[~frame.statistic.isin(["mean","std","min","max"])],frame[frame.statistic.isin(["mean","std","min","max"])]
def real_table():
    real,summary=source_real(); by_dataset={}
    for ds in DATASET_ORDER:
        by_dataset[ds]={}; sub=real[real.dataset==ds]
        for m in sub.method.unique():
            row=sub[sub.method==m].iloc[0]; by_dataset[ds][m]=(float(row.hard_f1),None if m=="constant_threshold" else float(row.hard_mae_log10_p))
        for m in ("gumbel99","mc99"):
            values=summary[(summary.dataset==ds)&(summary.method==m)]
            if len(values):
                row=values[values.statistic=="mean"].iloc[0]; by_dataset[ds][m]=(float(row.hard_f1),float(row.hard_mae_log10_p))
    rows=[]
    for dataset_index,ds in enumerate(DATASET_ORDER):
        methods=method_order(by_dataset[ds]); eligible=[m for m in methods if m!="gumbel999"]
        bf=max((by_dataset[ds][m][0] for m in eligible),default=np.nan); bm=min((by_dataset[ds][m][1] for m in eligible if by_dataset[ds][m][1] is not None),default=np.nan)
        rows.append({"raw_line":r"\multicolumn{4}{l}{\textbf{"+esc(DATASET_DISPLAY_NAMES.get(ds,ds))+r"}} "+r"\\","rule_before":r"\midrule" if dataset_index else None})
        previous_method_group=None
        for m in methods:
            fv,mv=by_dataset[ds][m]; ftxt,mtxt=fmt(fv,4 if fv>=0.9995 else 3),fmt(mv,4); bold=set()
            if m!="gumbel999":
                if np.isclose(fv,bf): bold.add("f1")
                if mv is not None and np.isclose(mv,bm): bold.add("mae")
            if m in ("gumbel99","mc99"):
                s=summary[(summary.dataset==ds)&(summary.method==m)&(summary.statistic=="std")].iloc[0]
                ftxt+=f" ({fmt(s.hard_f1,3)})"; mtxt+=f" ({fmt(s.hard_mae_log10_p,4)})"
            method_group=group_for(m)
            rule=r"\cmidrule(lr){2-4}" if previous_method_group is not None and method_group!=previous_method_group else None
            rows.append({"rule_before":rule,"columns":["dataset","method","f1","mae"],"display":{"dataset":"","method":esc(DISPLAY.get(m,m)),"f1":ftxt,"mae":mtxt},"bold":bold})
            previous_method_group=method_group
    write_table("real_hard_comparison.tex",["Dataset","Method","Hard F1",r"Hard MAE $-\log_{10}p$"],rows,"llrr")
def null_table():
    frame=pd.read_csv(TABLES/"null_calibration_summary.csv"); rows=[]
    for _,r in frame.iterrows():
        d={"method":esc(DISPLAY.get(r.method,r.method)),"n":str(int(r.n))}
        for a in ("0.01","0.05","0.10"): d[a]=f"{fmt(r[f'rejection_fraction_{a}'])} [{fmt(r[f'ci95_low_{a}'])}, {fmt(r[f'ci95_high_{a}'])}]"
        d["ks"]=fmt(r.ks_statistic if pd.notna(r.ks_statistic) else None); d["ks_p"]=fmt(r.ks_p_value if pd.notna(r.ks_p_value) else None)
        rows.append({"columns":["method","n","0.01","0.05","0.10","ks","ks_p"],"display":d})
    write_table("null_calibration.tex",["Method","$n$",r"Reject @ .01 (95\% CI)",r"Reject @ .05 (95\% CI)",r"Reject @ .10 (95\% CI)","KS $D$","KS $p$"],rows,"lrllllr")
    with (OUTPUT/"null_calibration.tex").open("a",encoding="utf-8") as h: h.write(r"\footnotesize{MC-99 p-values are discrete, so KS is not applicable; Gumbel-99 and MC-99 use the first 300 null datasets.}"+"\n")
def cost_table():
    cost=pd.read_csv(TABLES/"cost.csv"); frame=pd.read_csv(TABLES/"cost_by_n.csv"); rows=[]
    for m in ("mc999","mc99","gumbel99"):
        s=frame[frame.competitor==m]; d={"competitor":esc(DISPLAY[m]),"seconds":str(round(s.competitor_seconds.mean())),"xgb":fmt(cost.gumbel_pred_xgb_seconds_median5.mean(),3),"ridge":fmt(cost.gumbel_pred_ridge_seconds_median5.mean(),3)}; cols=["competitor","seconds","xgb","ridge"]
        for n in (100,315,450):
            sub=s[s.n_training_datasets==n]
            for mode in ("contended","calibrated"):
                key=f"{n}_{mode}"; d[key]=str(round(sub[f"break_even_datasets_{mode}"].mean())); cols.append(key)
        rows.append({"columns":cols,"display":d})
    headers=[r"Competitor & Competitor s & Surrogate XGBoost s & Surrogate ridge s & \multicolumn{2}{c}{$n = 100$} & \multicolumn{2}{c}{$n = 315$} & \multicolumn{2}{c}{$n = 450$} \\",r" & & & & Contended & Calibrated & Contended & Calibrated & Contended & Calibrated \\"]
    headers.insert(1,r"\cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}")
    write_table("cost_break_even.tex",[],rows,"lrrr"+"r"*6,headers)
def main():
    import argparse
    global REAL_SOURCE
    parser=argparse.ArgumentParser(); parser.add_argument("--real-source",choices=("original","reference"),default="original"); REAL_SOURCE=parser.parse_args().real_source
    synthetic_table(); real_table(); null_table(); cost_table()
    paired_bootstrap()
    for path in sorted(OUTPUT.glob("*.tex")): print(f"{path.name}: {path.stat().st_size} bytes")
if __name__=="__main__": main()
