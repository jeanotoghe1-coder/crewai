from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="ORUS Python Analytics", version="1.0.0")

MAX_ROWS = int(os.getenv("ORUS_PY_MAX_ROWS", "5000"))
API_KEY = os.getenv("ORUS_PY_ANALYTICS_KEY", "")
ALGORITHMS = {
    "forecast_timeseries", "anomaly_detection", "outlier_detection",
    "correlation_analysis", "monte_carlo_coverage", "leadtime_forecast",
    "trend_acceleration", "supplier_risk_scoring", "credit_risk_scoring",
}

class AnalyzeRequest(BaseModel):
    algorithm: str
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None

def require_key(key: Optional[str]) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid_analytics_key")

def finite(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def numeric_values(rows: List[Dict[str, Any]], field: str) -> np.ndarray:
    vals = [finite(r.get(field)) for r in rows]
    return np.array([x for x in vals if x is not None], dtype=float)

def robust_z(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad <= 1e-12:
        std = float(np.std(values))
        if std <= 1e-12:
            return np.zeros_like(values)
        return (values - float(np.mean(values))) / std
    return 0.6744897501960817 * (values - med) / mad

def forecast_timeseries(rows, p):
    field = str(p.get("value_field", "value")); horizon = max(1, min(int(p.get("horizon", 6)), 52))
    alpha = min(max(float(p.get("alpha", .45)), .05), .95); beta = min(max(float(p.get("beta", .20)), .01), .80)
    y = numeric_values(rows, field)
    if len(y) < 3: raise HTTPException(422, "forecast_requires_at_least_3_values")
    level=float(y[0]); trend=float(y[1]-y[0]); fitted=[level]
    for value in y[1:]:
        prev=level; level=alpha*float(value)+(1-alpha)*(level+trend); trend=beta*(level-prev)+(1-beta)*trend; fitted.append(level)
    resid=float(np.std(y-np.array(fitted,dtype=float))); forecast=[]
    for h in range(1,horizon+1):
        point=max(0.0, level+h*trend); sigma=resid*math.sqrt(h)
        forecast.append({"step":h,"forecast":round(point,6),"lower_95":round(max(0.0,point-1.96*sigma),6),"upper_95":round(point+1.96*sigma,6)})
    return {"method":"holt_linear","observations":len(y),"level":round(level,6),"trend":round(trend,6),"residual_std":round(resid,6),"forecast":forecast,"visualization_hint":"line"}

def anomaly_detection(rows,p):
    field=str(p.get("value_field","value")); threshold=min(max(float(p.get("threshold",3.5)),1.5),8.0); indexed=[]; values=[]
    for i,r in enumerate(rows):
        x=finite(r.get(field))
        if x is not None: indexed.append((i,r,x)); values.append(x)
    arr=np.array(values,dtype=float)
    if len(arr)<4: raise HTTPException(422,"anomaly_detection_requires_at_least_4_values")
    z=robust_z(arr); anomalies=[]
    for (i,row,value),score in zip(indexed,z):
        if abs(float(score))>=threshold: anomalies.append({"index":i,"value":value,"robust_z":round(float(score),6),"row":row})
    return {"method":"median_mad_robust_z","threshold":threshold,"observations":len(arr),"anomaly_count":len(anomalies),"anomalies":anomalies,"visualization_hint":"scatter"}

def outlier_detection(rows,p): return anomaly_detection(rows,p)

def correlation_analysis(rows,p):
    fields=p.get("fields")
    if not fields:
        c={}
        for r in rows[:50]:
            for k,v in r.items():
                if finite(v) is not None: c[k]=c.get(k,0)+1
        fields=[k for k,n in c.items() if n>=max(2,len(rows)//2)]
    fields=list(fields)[:12]
    if len(fields)<2: raise HTTPException(422,"correlation_requires_2_numeric_fields")
    mat=[]
    for r in rows:
        vals=[finite(r.get(f)) for f in fields]
        if all(v is not None for v in vals): mat.append(vals)
    if len(mat)<3: raise HTTPException(422,"correlation_requires_at_least_3_complete_rows")
    corr=np.nan_to_num(np.corrcoef(np.array(mat,dtype=float),rowvar=False),nan=0.0)
    return {"method":"pearson","fields":fields,"matrix":[[round(float(x),6) for x in row] for row in corr],"observations":len(mat),"visualization_hint":"heatmap"}

def monte_carlo_coverage(rows,p):
    sims=max(500,min(int(p.get("simulations",5000)),20000)); cv=min(max(float(p.get("demand_cv",.25)),.01),2.0); horizon=max(7,min(int(p.get("horizon_days",90)),365)); rng=np.random.default_rng(int(p.get("seed",42))); out=[]
    for row in rows[:200]:
        available=finite(row.get("available_quantity")); mean=finite(row.get("average_daily_consumption"))
        if available is None or mean is None or mean<=0: continue
        sigma2=math.log(1+cv**2); sigma=math.sqrt(sigma2); mu=math.log(max(mean,1e-9))-sigma2/2
        demand=rng.lognormal(mu,sigma,size=(sims,horizon)); cum=np.cumsum(demand,axis=1); hit=cum>=max(available,0.0); any_hit=hit.any(axis=1); first=np.where(any_hit,hit.argmax(axis=1)+1,horizon+1)
        out.append({"item_id":row.get("item_id"),"sku":row.get("sku"),"item_name":row.get("item_name"),"stockout_probability_7d":round(float(np.mean(first<=7)),6),"stockout_probability_30d":round(float(np.mean(first<=30)),6),"stockout_probability_horizon":round(float(np.mean(first<=horizon)),6),"stockout_day_p50":round(float(np.percentile(first,50)),2)})
    out.sort(key=lambda x:(-x["stockout_probability_30d"],x["stockout_day_p50"]))
    return {"method":"lognormal_monte_carlo","simulations":sims,"demand_cv":cv,"horizon_days":horizon,"items":out,"visualization_hint":"bar"}

def leadtime_forecast(rows,p):
    vals=numeric_values(rows,str(p.get("value_field","lead_days")))
    if len(vals)<3: raise HTTPException(422,"leadtime_forecast_requires_at_least_3_values")
    rng=np.random.default_rng(int(p.get("seed",42))); samples=max(500,min(int(p.get("bootstrap_samples",5000)),20000)); boot=rng.choice(vals,size=(samples,len(vals)),replace=True).mean(axis=1)
    return {"method":"bootstrap","observations":len(vals),"mean":round(float(np.mean(vals)),4),"median":round(float(np.median(vals)),4),"p90":round(float(np.percentile(vals,90)),4),"p95":round(float(np.percentile(vals,95)),4),"mean_ci95_low":round(float(np.percentile(boot,2.5)),4),"mean_ci95_high":round(float(np.percentile(boot,97.5)),4),"visualization_hint":"bar"}

def trend_acceleration(rows,p):
    y=numeric_values(rows,str(p.get("value_field","value")))
    if len(y)<3: raise HTTPException(422,"trend_acceleration_requires_at_least_3_values")
    g=[None if abs(a)<1e-12 else 100*(b-a)/abs(a) for a,b in zip(y[:-1],y[1:])]; valid=[x for x in g if x is not None]; acc=(valid[-1]-valid[-2]) if len(valid)>=2 else None
    return {"method":"period_growth_acceleration","growth_pct":[None if x is None else round(float(x),4) for x in g],"latest_acceleration_pp":None if acc is None else round(float(acc),4),"direction":"accelerating" if acc and acc>0 else "decelerating" if acc and acc<0 else "stable","visualization_hint":"line"}

def supplier_risk_scoring(rows,p):
    out=[]
    for r in rows:
        overdue=max(0.,finite(r.get("overdue_orders")) or 0.); orders=max(1.,finite(r.get("orders_count")) or 1.); otif=finite(r.get("on_time_delivery_pct")); quality=finite(r.get("quality_acceptance_pct")); returns=finite(r.get("return_rate_pct"))
        score=100*(.40*min(overdue/orders,1)+.30*max(0,1-(otif if otif is not None else 100)/100)+.20*max(0,1-(quality if quality is not None else 100)/100)+.10*min(max((returns or 0)/100,0),1))
        out.append({"supplier_id":r.get("supplier_id"),"supplier_name":r.get("supplier_name"),"risk_score":round(score,2),"risk_level":"high" if score>=60 else "medium" if score>=30 else "low"})
    out.sort(key=lambda x:x["risk_score"],reverse=True); return {"method":"weighted_supplier_risk","suppliers":out,"visualization_hint":"bar"}

def credit_risk_scoring(rows,p):
    out=[]
    for r in rows:
        amount=max(0.,finite(r.get("amount_base")) or finite(r.get("exposure")) or 0.); limit_=finite(r.get("credit_limit")) or finite(r.get("limit_base")); overdue=max(0.,finite(r.get("overdue_amount")) or 0.); util=(amount/limit_) if limit_ and limit_>0 else (1. if amount>0 else 0.); ratio=overdue/max(amount,1.); score=100*min(1,.75*min(util,1.5)/1.5+.25*min(ratio,1))
        out.append({"customer_id":r.get("customer_id"),"customer_name":r.get("customer_name"),"exposure":amount,"utilization_ratio":round(util,4),"risk_score":round(score,2),"risk_level":"high" if score>=70 else "medium" if score>=40 else "low"})
    out.sort(key=lambda x:x["risk_score"],reverse=True); return {"method":"weighted_credit_risk","customers":out,"visualization_hint":"bar"}

DISPATCH={"forecast_timeseries":forecast_timeseries,"anomaly_detection":anomaly_detection,"outlier_detection":outlier_detection,"correlation_analysis":correlation_analysis,"monte_carlo_coverage":monte_carlo_coverage,"leadtime_forecast":leadtime_forecast,"trend_acceleration":trend_acceleration,"supplier_risk_scoring":supplier_risk_scoring,"credit_risk_scoring":credit_risk_scoring}

@app.get("/health")
def health(): return {"service":"ORUS Python Analytics","version":"1.0.0","algorithms":sorted(ALGORITHMS),"max_rows":MAX_ROWS}

@app.post("/analyze")
def analyze(req:AnalyzeRequest,x_orus_analytics_key:Optional[str]=Header(default=None)):
    require_key(x_orus_analytics_key)
    if req.algorithm not in ALGORITHMS: raise HTTPException(400,"algorithm_not_allowed")
    if len(req.rows)>MAX_ROWS: raise HTTPException(413,"dataset_too_large")
    return {"ok":True,"trace_id":req.trace_id,"algorithm":req.algorithm,"rows_received":len(req.rows),"result":DISPATCH[req.algorithm](req.rows,req.params)}
