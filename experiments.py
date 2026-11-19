"""Reproducible benchmark runner. Results are optimization fitness, not ML accuracy."""
import argparse, csv
from kolamnet import evolve_ga, hill_climb, random_search, make_target

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--grid",type=int,default=8,choices=[8,12,16])
    ap.add_argument("--runs",type=int,default=5)
    ap.add_argument("--generations",type=int,default=80)
    ap.add_argument("--population",type=int,default=80)
    ap.add_argument("--task",choices=["design","recreation"],default="design")
    ap.add_argument("--output",default="benchmark_results.csv")
    args=ap.parse_args()
    rows=[]
    target=make_target(args.grid) if args.task=="recreation" else None
    budget=args.population*args.generations
    for seed in range(args.runs):
        results=[
            ("GA",evolve_ga(args.grid,args.task,target,args.population,args.generations,0.03,seed)),
            ("Hill climbing",hill_climb(args.grid,args.task,target,budget,seed)),
            ("Random search",random_search(args.grid,args.task,target,budget,seed))
        ]
        for name,res in results:
            rows.append({"seed":seed,"grid":args.grid,"task":args.task,"algorithm":name,"fitness":res["fitness"]})
            print(f"seed={seed:02d} {name:14s} fitness={res['fitness']:.4f}")
    with open(args.output,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\nSaved {args.output}")
    for name in sorted(set(r["algorithm"] for r in rows)):
        vals=[r["fitness"] for r in rows if r["algorithm"]==name]
        print(f"{name:14s} mean={sum(vals)/len(vals):.4f}")

if __name__=="__main__": main()
