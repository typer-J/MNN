#!/usr/bin/env python3
"""Isolated FP32 layout exploration. No production dispatch modifications."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def extract(text, name):
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    mask = re.sub(pattern, lambda m: ' ' * len(m.group()), text)
    hits = list(re.finditer(r'void\s+'+name+r'\s*\([^;{}]*\)\s*\{',mask))
    if len(hits)!=1: raise RuntimeError('ambiguous scalar definition')
    start=hits[0].start(); end=hits[0].end(); depth=1
    while depth:
        depth+=(mask[end]=='{')-(mask[end]=='}'); end+=1
    return text[start:end]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=8)
    args=parser.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    if platform.machine()!='riscv64': raise RuntimeError('native RISC-V required')
    os.sched_setaffinity(0,{args.cpu})
    meta={'status':'started','machine':platform.machine(),'kernel':platform.release(),
          'affinity':sorted(os.sched_getaffinity(0)),'commands':[], 'source_hashes':{},
          'scope':'FP32 synthetic direct calls, single thread, no MNN graph dispatch or quantized LLM claim',
          'compiler':subprocess.check_output(['g++','--version'],text=True),
          'load_before':os.getloadavg()}
    def save(): (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    def cmd(argv,log):
        meta['commands'].append(argv);save()
        with (out/log).open('w') as f: subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        root=args.source.resolve(); snap=out/'source'; snap.mkdir()
        paths=['source/backend/cpu/compute/CommonOptFunction.cpp','source/backend/cpu/compute/CommonOptFunction.h',
               'source/math/Vec.hpp','source/core/Macro.h','source/core/SimdHeader.h','include/MNN/MNNDefine.h',
               'source/backend/cpu/riscv/rvv/MNNPackedMatMulRemainFP32.cpp']
        for rel in paths:
            dest=snap/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,dest)
            meta['source_hashes'][rel]=sha(dest)
        here=Path(__file__).resolve().parent
        for name in ['run.py','benchmark.cpp','kernels.cpp','probe.cpp']:
            shutil.copyfile(here/name,out/name);meta['source_hashes'][name]=sha(out/name)
        common=(snap/paths[0]).read_text()
        header=(snap/paths[1]).read_text()
        struct=re.search(r'struct MatMulParam\s*\{[^}]*\};',header).group()
        prototype='void MNNComputeMatMulForE_1(const float*, const float*, float*, const float*, const MatMulParam*, size_t);\n'
        (out/'baseline.hpp').write_text('#include <cstdint>\n#include <cstddef>\n'+struct+'\n'+prototype)
        fragment=extract(common,'MNNComputeMatMulForE_1')
        meta['exact_scalar_fragment_sha256']=hashlib.sha256(fragment.encode()).hexdigest()
        (out/'baseline.cpp').write_text('#include "baseline.hpp"\n#include "math/Vec.hpp"\nusing Vec4=MNN::Math::Vec<float,4>;\n'+fragment+'\n')
        flags=['g++','-std=c++11','-O3','-fno-fast-math','-ffp-contract=off','-fno-lto','-fno-exceptions',
               '-fno-rtti','-fno-tree-vectorize','-fno-tree-slp-vectorize','-mabi=lp64d',
               '-I'+str(snap/'source'),'-I'+str(snap/'include'),'-I'+str(out)]
        cmd(flags+['-march=rv64gcv',str(out/'probe.cpp'),'-o',str(out/'probe')],'probe-build.log')
        cmd([str(out/'probe')],'probe.json')
        meta['probe']=json.loads((out/'probe.json').read_text())
        if not meta['probe']['vector_fp32_probe_passed']: raise RuntimeError('RVV FP32 probe failed')
        cmd(flags+['-march=rv64gc','-c',str(out/'baseline.cpp'),'-o',str(out/'baseline.o')],'baseline-build.log')
        for mode in ['emulated','native']:
            extra=['-DEMULATE','-march=rv64gc'] if mode=='emulated' else ['-march=rv64gcv']
            obj=out/(mode+'.o')
            cmd(flags+extra+['-c',str(out/'kernels.cpp'),'-o',str(obj)],mode+'-kernels-build.log')
            inputs=[str(out/'benchmark.cpp'),str(out/'baseline.o'),str(obj)]
            if mode=='native':
                current_flags=[f for f in flags if f not in ['-fno-tree-vectorize','-fno-tree-slp-vectorize']]
                cmd(current_flags+['-march=rv64gcv','-c',str(snap/paths[-1]),'-o',str(out/'current-c4.o')],'current-c4-build.log')
                inputs.append(str(out/'current-c4.o'))
            binary=out/mode
            cmd(flags+['-march=rv64gc']+(['-DEMULATE'] if mode=='emulated' else [])+inputs+['-o',str(binary)],mode+'-link.log')
            cmd([str(binary),'--correctness-only'],mode+'-correctness.log')
        for index in range(1,4):
            print('performance',index,flush=True)
            with (out/f'samples-{index}.csv').open('w') as stdout, (out/f'performance-{index}.log').open('w') as stderr:
                subprocess.run([str(out/'native')],stdout=stdout,stderr=stderr,check=True)
        meta['status']='passed';meta['load_after']=os.getloadavg()
        meta['artifact_hashes']={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file() and p.name!='metadata.json'}
    except Exception as error:
        meta['status']='failed';meta['error']=str(error);raise
    finally: save()

if __name__=='__main__': main()
