#!/usr/bin/env python3
import argparse, collections, math

def ngrams(s, n):
    return [s[i:i+n] for i in range(len(s)-n+1)] if len(s) >= n else []

def trigram_precision(gen_text, ref_text, n=3):
    g = ngrams(gen_text, n); r = ngrams(ref_text, n)
    if not g: return 0.0
    cg, cr = collections.Counter(g), collections.Counter(r)
    return sum(min(cg[t], cr.get(t,0)) for t in cg)/len(g)

def trigram_repetition(gen_text, n=3):
    g = ngrams(gen_text, n); 
    if not g: return 0.0, 0.0
    c = collections.Counter(g); total = len(g)
    rep_share  = sum(max(v-1,0) for v in c.values())/total
    rep_unique = sum(1 for v in c.values() if v>1)/total
    return rep_share, rep_unique

def read(paths):
    import pathlib
    return "\n".join(pathlib.Path(p).read_text(encoding='utf-8', errors='ignore') for p in paths)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    ref = read(args.ref)
    gen = read(args.gen)
    p = trigram_precision(gen, ref, args.n)
    rs, ru = trigram_repetition(gen, args.n)
    print(f"n={args.n}")
    print(f"Trigram precision (specific): {p:.6f}")
    print(f"Trigram repetition share (general): {rs:.6f}")
    print(f"Trigram repetition unique (general): {ru:.6f}")