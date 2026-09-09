#!/usr/bin/env python3
"""Verify three native process records and compare identical shapes/phases."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.results;out=args.out;out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((root/'metadata.json').read_text())
    assert meta['status']=='passed' and meta['machine']=='riscv64'
    for name,digest in meta['artifact_hashes'].items():
        rel=Path(name)
        assert not rel.is_absolute() and len(rel.parts)==1 and ':' not in name and '\\' not in name
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,name
    all_runs=[]
    for run in range(1,4):
        assert 'status=passed' in (root/f'performance-{run}.log').read_text()
        groups={}
        for row in csv.DictReader((root/f'samples-{run}.csv').open()):
            key=tuple(row[x] for x in ('e','k','h','transpose','bias','variant','phase'))
            ns=float(row['ns']);assert math.isfinite(ns) and ns>0 and int(row['iterations'])>0
            groups.setdefault(key,[]).append((int(row['round']),ns))
        for key,values in groups.items(): assert sorted(r for r,n in values)==list(range(7)),key
        all_runs.append({key:st.median(n for r,n in values) for key,values in groups.items()})
    assert all_runs[0].keys()==all_runs[1].keys()==all_runs[2].keys()
    records=[]
    for key in sorted(all_runs[0]):
        ns=[r[key] for r in all_runs];e,k,h,t,b,variant,phase=key
        basevariant='original_e1' if phase=='compute' else 'panel4'
        basekey=key[:5]+(basevariant,phase)
        if basekey not in all_runs[0]: continue
        ratios=[r[basekey]/r[key] for r in all_runs]
        currentkey=key[:5]+('current_c4','compute')
        current=[r[currentkey]/r[key] for r in all_runs] if phase=='compute' and currentkey in all_runs[0] else []
        records.append(dict(e=int(e),k=int(k),h=int(h),transpose=int(t),bias=int(b),variant=variant,phase=phase,
                            median_ns=st.median(ns),process_min_ns=min(ns),process_max_ns=max(ns),
                            baseline=basevariant,speedup=st.median(ratios),all_gain_5=min(ratios)>1.05,
                            all_loss_5=max(ratios)<1/1.05,current_c4_speedup=st.median(current) if current else ''))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,records[0].keys());w.writeheader();w.writerows(records)
    summary={}
    for phase in ['compute','weight_pack','steady_chain','first_chain']:
        for variant in ['direct_rvv','panel4','panel8','panel16']:
            for regime in ['e1','e_gt_1']:
                rows=[r for r in records if r['phase']==phase and r['variant']==variant and (r['e']==1)==(regime=='e1')]
                if rows:
                    summary[f'{regime}/{phase}/{variant}']={'cases':len(rows),
                        'geomean_speedup':math.exp(st.mean(math.log(r['speedup']) for r in rows)),
                        'all_process_gain_5':sum(r['all_gain_5'] for r in rows),
                        'all_process_loss_5':sum(r['all_loss_5'] for r in rows)}
    # Pack payback for repeated GEMV calls; measured pack plus compute, excluding graph costs.
    amort=[]
    for r in records:
        if r['e']!=1 or r['phase']!='compute' or r['variant'] not in ['panel4','panel8','panel16']:continue
        key=tuple(str(r[x]) for x in ['e','k','h','transpose','bias'])
        variant=r['variant']
        for baseline in ['original_e1']+(['direct_rvv'] if not r['transpose'] else []):
            breaks=[];ratios={n:[] for n in (1,8,32,128)}
            for run in all_runs:
                base=run[key+(baseline,'compute')];comp=run[key+(variant,'compute')];pack=run[key+(variant,'weight_pack')]
                breaks.append(math.ceil(pack/(base-comp)) if base>comp else None)
                for n in ratios:ratios[n].append(n*base/(pack+n*comp))
            amort.append({**{x:r[x] for x in ['e','k','h','transpose','bias','variant']},'baseline':baseline,
                          'break_even_reuses_each_process':breaks,
                          'speedup_at_reuses':{n:st.median(v) for n,v in ratios.items()}})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'amortization.json').write_text(json.dumps(amort,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
