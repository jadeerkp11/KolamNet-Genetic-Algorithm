"""Core KolamNet optimization and image-to-tile conversion."""
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# 4-bit side masks: N=1, E=2, S=4, W=8.
TILE_MASKS = np.array([0, 3, 5, 6, 9, 10, 12, 15], dtype=np.uint8)

def mirror_lr(p):
    lut = np.zeros(16, dtype=np.uint8)
    for m in range(16):
        lut[m] = (m & 5) | ((m & 2) << 2) | ((m & 8) >> 2)
    return lut[p[:, ::-1]]

def mirror_tb(p):
    lut = np.zeros(16, dtype=np.uint8)
    for m in range(16):
        lut[m] = (m & 10) | ((m & 1) << 2) | ((m & 4) >> 2)
    return lut[p[::-1, :]]

def decode_symmetric(genes, n):
    q = np.asarray(genes, dtype=np.uint8).reshape(n//2, n//2)
    top = np.concatenate([q, mirror_lr(q)], axis=1)
    return np.concatenate([top, mirror_tb(top)], axis=0)

def random_genes(n, rng):
    return rng.choice(TILE_MASKS, size=(n//2, n//2)).astype(np.uint8)

def _edges(pattern):
    p = pattern
    dangling = 0
    drawn_edges = 0
    # East-West and South-North neighbor agreement.
    for r in range(p.shape[0]):
        for c in range(p.shape[1]):
            m = int(p[r,c])
            for bit in (1,2,4,8):
                if m & bit:
                    drawn_edges += 1
            if c == p.shape[1]-1 and (m & 2): dangling += 1
            if r == p.shape[0]-1 and (m & 4): dangling += 1
            if c == 0 and (m & 8): dangling += 1
            if r == 0 and (m & 1): dangling += 1
            if c < p.shape[1]-1 and bool(m & 2) != bool(int(p[r,c+1]) & 8): dangling += 1
            if r < p.shape[0]-1 and bool(m & 4) != bool(int(p[r+1,c]) & 1): dangling += 1
    return dangling, max(1, drawn_edges)

def continuity(pattern):
    d, total = _edges(pattern)
    return max(0.0, 1.0 - d / (2 * pattern.size + 2 * pattern.shape[0] + 2 * pattern.shape[1]))

def components_info(pattern):
    n = pattern.shape[0]
    seen = set()
    closed_cells = 0
    loops = 0
    drawn = int(np.count_nonzero(pattern))
    for r in range(n):
        for c in range(n):
            if pattern[r,c] == 0 or (r,c) in seen: continue
            stack=[(r,c)]; seen.add((r,c)); comp=[]; closed=True
            while stack:
                y,x=stack.pop(); comp.append((y,x)); m=int(pattern[y,x])
                checks=[(1,0,4,1),(0,1,2,8),(-1,0,1,4),(0,-1,8,2)]
                for dy,dx,b1,b2 in checks:
                    if not (m & b1): continue
                    yy,xx=y+dy,x+dx
                    if not (0<=yy<n and 0<=xx<n) or not (int(pattern[yy,xx]) & b2):
                        closed=False
                    elif (yy,xx) not in seen:
                        seen.add((yy,xx)); stack.append((yy,xx))
                # Any unmatched side makes component open.
                if any((m & bit) and not _side_matched(pattern,y,x,bit) for bit in (1,2,4,8)):
                    closed=False
            if closed and len(comp)>=4:
                closed_cells += len(comp); loops += 1
    closure = closed_cells / max(1, drawn)
    loop_score = min(1.0, loops / max(1, n//3))
    ink = np.count_nonzero(pattern) / pattern.size
    ink_score = max(0.0, 1.0 - abs(ink - 0.45) / 0.45)
    sym = (np.mean(pattern == mirror_lr(pattern)) + np.mean(pattern == mirror_tb(pattern))) / 2
    return {"closure":float(closure), "loops":float(loop_score), "ink":float(ink_score),
            "symmetry":float(sym), "ink_fraction":float(ink), "loop_count":int(loops)}

def _side_matched(p,r,c,bit):
    n=p.shape[0]; m=int(p[r,c])
    if bit==1: return r>0 and bool(int(p[r-1,c])&4)
    if bit==2: return c<n-1 and bool(int(p[r,c+1])&8)
    if bit==4: return r<n-1 and bool(int(p[r+1,c])&1)
    return c>0 and bool(int(p[r,c-1])&2)

def fitness_design(pattern):
    x=components_info(pattern)
    return 0.30*x["closure"]+0.25*x["loops"]+0.20*x["ink"]+0.25*x["symmetry"]

def fitness_recreation(pattern, target, creativity=0.0):
    """Original: 0.7*similarity + 0.3*continuity.
    creativity (0..1) optionally blends in the generative-design fitness so that
    recreations are inspired by the photo instead of copying it. 0.0 = original."""
    similarity=float(np.mean(pattern==target))
    base=0.7*similarity+0.3*continuity(pattern)
    if creativity<=0: return base
    return (1-creativity)*base+creativity*fitness_design(pattern)

def fitness(pattern, task, target=None, creativity=0.0):
    if task=="design": return fitness_design(pattern)
    return fitness_recreation(pattern,target,creativity)

def _block_crossover(a,b,rng):
    child=a.copy()
    h,w=a.shape
    y1,y2=sorted(rng.integers(0,h+1,size=2))
    x1,x2=sorted(rng.integers(0,w+1,size=2))
    if y1==y2: y2=min(h,y1+1)
    if x1==x2: x2=min(w,x1+1)
    child[y1:y2,x1:x2]=b[y1:y2,x1:x2]
    return child

def evolve_ga(n=12, task="design", target=None, population_size=100, generations=100,
              mutation_rate=0.03, seed=42, evaluations=None, creativity=0.0):
    rng=np.random.default_rng(seed)
    genes_shape=(n//2,n//2)
    pop=[random_genes(n,rng) for _ in range(population_size)]
    def decode(g): return decode_symmetric(g,n)
    scores=[fitness(decode(g),task,target,creativity) for g in pop]
    best_idx=int(np.argmax(scores)); best=pop[best_idx].copy(); best_score=scores[best_idx]
    for _ in range(generations):
        new=[best.copy()]
        if len(pop)>1: new.append(pop[int(np.argmax(scores))].copy())
        while len(new)<population_size:
            def tournament():
                ids=rng.integers(0,len(pop),size=3)
                return pop[int(ids[np.argmax([scores[i] for i in ids])])]
            p1,p2=tournament(),tournament()
            child=_block_crossover(p1,p2,rng) if rng.random()<0.8 else p1.copy()
            mask=rng.random(genes_shape)<mutation_rate
            if np.any(mask): child[mask]=rng.choice(TILE_MASKS,size=int(mask.sum()))
            new.append(child)
        pop=new
        scores=[fitness(decode(g),task,target,creativity) for g in pop]
        i=int(np.argmax(scores))
        if scores[i]>best_score: best,best_score=pop[i].copy(),scores[i]
    pat=decode(best)
    return {"pattern":pat,"fitness":float(best_score),"components":components_info(pat),"seed":int(seed)}

def hill_climb(n=12, task="design", target=None, evaluations=10000, seed=42):
    rng=np.random.default_rng(seed); g=random_genes(n,rng)
    def score(x): return fitness(decode_symmetric(x,n),task,target)
    s=score(g); best=g.copy(); bs=s
    for _ in range(max(1,evaluations-1)):
        cand=best.copy(); y=int(rng.integers(0,n//2)); x=int(rng.integers(0,n//2))
        cand[y,x]=rng.choice(TILE_MASKS); cs=score(cand)
        if cs>=bs: best,bs=cand,cs
    p=decode_symmetric(best,n)
    return {"pattern":p,"fitness":float(bs),"components":components_info(p)}

def random_search(n=12, task="design", target=None, evaluations=10000, seed=42):
    rng=np.random.default_rng(seed); best=None; bs=-1
    for _ in range(max(1,evaluations)):
        g=random_genes(n,rng); p=decode_symmetric(g,n); s=fitness(p,task,target)
        if s>bs: best,bs=p,s
    return {"pattern":best,"fitness":float(bs),"components":components_info(best)}

def make_target(n, kind="rings"):
    p=np.zeros((n,n),dtype=np.uint8)
    # Simple symmetric closed tile motifs used for reproducible synthetic tests.
    # Build rectangular loop boundaries using side-paired tiles.
    if kind=="rings":
        levels=[max(1,n//4-1), max(2,n//2-1)]
        for rad in levels:
            lo=(n-rad)//2; hi=lo+rad-1
            if hi<=lo: continue
            for c in range(lo,hi+1):
                p[lo,c] |= 6 if c==lo else (12 if c==hi else 10)
                p[hi,c] |= 3 if c==lo else (9 if c==hi else 10)
            for r in range(lo+1,hi):
                p[r,lo] |= 5
                p[r,hi] |= 5
    # Keep only allowed tile masks; nearest valid mask.
    return np.array([TILE_MASKS[np.argmin(np.abs(TILE_MASKS-int(v)))] for v in p.flat],dtype=np.uint8).reshape(n,n)

def otsu_threshold(gray):
    """Classic Otsu threshold on a uint8 array (pure numpy)."""
    hist=np.bincount(gray.ravel(),minlength=256).astype(float)
    total=hist.sum(); sum_all=(np.arange(256)*hist).sum()
    w0=0.0; sum0=0.0; best=0.0; thr=125
    for t in range(256):
        w0+=hist[t]
        if w0==0: continue
        w1=total-w0
        if w1==0: break
        sum0+=t*hist[t]
        m0=sum0/w0; m1=(sum_all-sum0)/w1
        var=w0*w1*(m0-m1)**2
        if var>best: best=var; thr=t
    return int(thr)

def _thin_strokes(mask, size):
    """Keep only thin strokes: remove anything wide enough to survive an
    erosion+dilation (morphological opening) of the given odd size, i.e. doors,
    big flowers and filled shapes. mask is a boolean array."""
    m=Image.fromarray((mask*255).astype(np.uint8))
    opened=m.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    return mask & ~(np.asarray(opened)>127)

def suggest_photo_settings(image):
    """Heuristic (NOT a trained model): guess line polarity and where the line
    pattern sits. Lines are assumed to be the minority class after Otsu
    thresholding, so white rice-flour kolam on a dark floor is handled too.
    Returns dict(crop_box=(l,t,r,b) as fractions, invert=bool, threshold=int)."""
    im=ImageOps.autocontrast(ImageOps.grayscale(image))
    scale=320/max(im.size)
    if scale<1: im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))))
    arr=np.asarray(im)
    t=otsu_threshold(arr)
    dark=arr<=t
    invert=bool(dark.mean()>0.5)            # lines are the minority class
    mask=(~dark) if invert else dark
    m=Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3))
    mask=np.asarray(m)>127
    mask=_thin_strokes(mask,7)              # ignore doors / big flowers / filled blobs
    ys,xs=np.nonzero(mask)
    if len(xs)<20:
        return {"crop_box":(0.0,0.0,1.0,1.0),"invert":invert,"threshold":125}
    h,w=mask.shape
    x1,x2=np.percentile(xs,[2,98]); y1,y2=np.percentile(ys,[2,98])
    pad_x=0.05*(x2-x1)+2; pad_y=0.05*(y2-y1)+2
    x1,x2,y1,y2=x1-pad_x,x2+pad_x,y1-pad_y,y2+pad_y
    side=max(x2-x1,y2-y1); cx=(x1+x2)/2; cy=(y1+y2)/2
    x1,x2,y1,y2=cx-side/2,cx+side/2,cy-side/2,cy+side/2
    box=(max(0,x1)/w,max(0,y1)/h,min(w,x2)/w,min(h,y2)/h)
    work=ImageOps.invert(Image.fromarray(arr)) if invert else Image.fromarray(arr)
    thr=otsu_threshold(np.asarray(work))
    return {"crop_box":tuple(float(v) for v in box),"invert":invert,"threshold":int(thr)}

def image_to_target(image, n=12, threshold=125, crop_box=None, invert=False, auto_threshold=False, ignore_blobs=False):
    """Approximate tile masks by sampling dark pixels at each cell side midpoint.
    Optional: crop_box=(l,t,r,b) fractions of the image, invert=True for light lines
    on dark ground, auto_threshold=True to use Otsu, ignore_blobs=True to drop thick solid areas
    (doors, big flowers) and keep thin strokes. Defaults == original behaviour."""
    if crop_box is not None:
        l,t,r,b=crop_box; W,H=image.size
        box=(int(l*W),int(t*H),max(int(l*W)+2,int(r*W)),max(int(t*H)+2,int(b*H)))
        image=image.crop(box)
    im=ImageOps.grayscale(image)
    im=ImageOps.autocontrast(im)
    if invert: im=ImageOps.invert(im)
    if auto_threshold: threshold=otsu_threshold(np.asarray(im))
    side=max(im.size)
    canvas=Image.new("L",(side,side),255)
    canvas.paste(im,((side-im.width)//2,(side-im.height)//2))
    canvas=canvas.resize((n*40,n*40),Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(0.6))
    arr=np.asarray(canvas)
    binary=arr<threshold
    if ignore_blobs: binary=_thin_strokes(binary,2*max(2,int(n*40/60))+1)
    cell=40
    pat=np.zeros((n,n),dtype=np.uint8)
    offsets={"N":(3,20),"E":(20,37),"S":(37,20),"W":(20,3)}
    bits={"N":1,"E":2,"S":4,"W":8}
    for r in range(n):
        for c in range(n):
            mask=0
            for side_name,(dy,dx) in offsets.items():
                patch=binary[r*cell+dy-2:r*cell+dy+3,c*cell+dx-2:c*cell+dx+3]
                if patch.mean()>0.18: mask|=bits[side_name]
            pat[r,c]=TILE_MASKS[np.argmin(np.abs(TILE_MASKS-mask))]
    preview=Image.fromarray(np.where(binary,0,255).astype(np.uint8)).resize((360,360))
    return pat,preview

def render_pattern(pattern, ax=None):
    import matplotlib.pyplot as plt
    if ax is None: _,ax=plt.subplots(figsize=(6,6))
    n=pattern.shape[0]
    ax.set_aspect("equal"); ax.set_xlim(-0.5,n-0.5); ax.set_ylim(n-0.5,-0.5)
    ax.set_xticks([]); ax.set_yticks([])
    # Pulli grid
    for r in range(n+1):
        for c in range(n+1):
            ax.plot(c-0.5,r-0.5,".",color="black",markersize=2.3,alpha=.65)
    # Draw each cell's tile as a smooth-ish connected path through side midpoints.
    pts={1:(0.5,0),2:(1,0.5),4:(0.5,1),8:(0,0.5)}
    for r in range(n):
        for c in range(n):
            m=int(pattern[r,c]); cx=c-0.5; cy=r-0.5
            pairs=[]
            if m==3: pairs=[(1,2)]
            elif m==5: pairs=[(1,4)]
            elif m==6: pairs=[(2,4)]
            elif m==9: pairs=[(1,8)]
            elif m==10: pairs=[(2,8)]
            elif m==12: pairs=[(4,8)]
            elif m==15: pairs=[(1,4),(2,8)]
            for a,b in pairs:
                x1,y1=pts[a]; x2,y2=pts[b]
                ax.plot([cx+x1,cx+x2],[cy+y1,cy+y2],color="black",lw=1.8,solid_capstyle="round")
    return ax
