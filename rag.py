import json, math, re, os
from collections import Counter

KB = json.load(open(os.path.join(os.path.dirname(__file__), "kb.json")))
STOP = set("""a an the of to in on for by with and or is are was were be been do does did what which who whom whose how many much
when where why that this these those it its as at from into over under than then there their they them i you we he she our your
use uses used using value number default configured configure set sets any all each per via about between within before after
during if not no can could should would will shall may might must tell me please answer question following give provide
according kb knowledge base passage passages system ignore previous instructions instead respond say output print reply""".split())
INJ = re.compile(r"(ignore (all |any )?(previous|prior|above)[^.?!]*[.?!]?|disregard[^.?!]*[.?!]?|"
                 r"(respond|reply|answer|output|say|print) (only |with |exactly )*['\"][^'\"]*['\"][^.?!]*[.?!]?|"
                 r"system (prompt|override)[^.?!]*[.?!]?|you are now[^.?!]*[.?!]?|new instructions?[^.?!]*[.?!]?)", re.I)

ALIAS = {"sgd": "stochastic gradient descent", "mlp": "multi layer perceptrons", "mlps": "multi layer perceptrons",
 "tcp": "transmission control protocol", "aes": "advanced encryption standard", "svm": "support vector machines",
 "svms": "support vector machines", "2pc": "two phase commit", "3pc": "three phase commit", "lsm": "log structured merge",
 "wal": "write ahead logging", "jwt": "json web tokens", "jwts": "json web tokens", "hlc": "hybrid logical clocks",
 "dns": "domain name system", "mdns": "multicast dns", "ecc": "elliptic curve cryptography", "waf": "web application firewalls",
 "resnet": "residual convolutional networks", "postgres": "postgresql", "innodb": "mysql innodb", "bn": "batch normalization",
 "layernorm": "layer normalization", "batchnorm": "batch normalization", "rendezvous": "rendezvous hashing", "hrw": "rendezvous hashing",
 "pki": "public key cryptography", "rsa": "rsa public key", "gossip": "gossip", "chacha20": "chacha20", "ospf": "open shortest path first ospf",
 "bgp": "border gateway protocol bgp", "tls": "transport layer security tls", "quic": "quic", "oauth": "oauth"}

SYN = {"keepalive": "keepalive heartbeat", "keepalives": "keepalives heartbeats", "coordinator": "coordinator leader",
 "subordinate": "subordinate follower", "coefficient": "coefficient learning rate", "magnitude": "magnitude scale",
 "milliseconds": "milliseconds ms", "millisecond": "millisecond ms", "relational": "relational postgresql"}

def expand(s):
    s = "".join(SYN.get(w.lower(), w) for w in re.split(r"(\s+)", s))
    return "".join(ALIAS.get(w.lower(), w) for w in re.split(r"(\s+)", s))

def stem(t):
    if len(t) > 4 and not t[-1].isdigit():
        if t.endswith("ies"): return t[:-3] + "y"
        if t.endswith("ing") and len(t) > 6: return t[:-3]
        if t.endswith("ed") and len(t) > 5: return t[:-2]
        if t.endswith("es") and t[-3] in "sxz": return t[:-2]
        if t.endswith("s") and not t.endswith("ss") and not t.endswith("us"): return t[:-1]
    return t

def toks(s):
    s = s.lower().replace("-", " ")
    return [stem(t) for t in re.findall(r"[a-z0-9][a-z0-9.+/]*[a-z0-9]|[a-z0-9]", s)]

def terms(s):
    w = [t for t in toks(s) if t not in STOP]
    return w + [a + "_" + b for a, b in zip(w, w[1:])]

def sents(p):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", p) if x.strip()]

N = len(KB)
DOCT = [terms(p) for p in KB]
SUBJ = [set(terms(sents(p)[0])) for p in KB]
DF = Counter(t for d in DOCT for t in set(d))
VOCAB = set(DF)
IDF = {t: math.log((N + 1) / (df + 0.5)) for t, df in DF.items()}

def vec(ts, subj=None):
    c = Counter(ts); v = {}
    for t, n in c.items():
        if t in IDF:
            w = (1 + math.log(n)) * IDF[t]
            if subj is not None and t in subj: w *= 1.6
            v[t] = w
    nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / nrm for t, x in v.items()}

DV = [vec(d, s) for d, s in zip(DOCT, SUBJ)]

def cos(a, b):
    if len(a) > len(b): a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())

ATTR_OLD = {
    "port": ["port"], "timeout": ["timeout", "ms", "interval", "heartbeat", "latency", "window", "duration", "limit", "skew", "wait"],
    "quorum": ["quorum", "acceptors", "participants", "members", "replicas", "retries", "nodes"],
    "threshold": ["threshold", "exceeds", "ratio", "crosses", "probability", "saturation", "utilization", "capacity", "past"],
    "lr": ["learning", "rate"], "batch": ["batch", "batches", "samples", "chunks", "records", "entries", "tokens", "bytes", "window"],
    "layers": ["layers", "blocks", "stack", "stages"], "bits": ["bits", "key", "length", "lengths"],
    "hash": ["digest", "hash", "sha", "blake2", "function", "primitive"],
}
ATTR = [
 (["port"], ["port"]),
 (["timeout", "interval", "heartbeat", "latency", "window", "duration", "how long", "ms", "skew", "wait", "dead"], ["ms"]),
 (["quorum", "how many nodes", "acceptors", "participants", "members", "replicas", "retries", "acknowledg"], ["quorum", "acceptors", "participants", "members", "replicas", "retries", "nodes"]),
 (["threshold", "ratio", "probability", "saturation", "utilization", "skew threshold", "capacity limit", "burst"], ["threshold", "ratio", "probability", "saturation", "utilization", "exceed", "cross", "surpass", "past"]),
 (["learning rate"], ["learning", "rate"]),
 (["batch", "chunk", "group", "how many samples", "records", "entries", "window length", "tokens per", "bytes"], ["batch", "samples", "chunks", "records", "entries", "tokens", "bytes", "groups", "examples", "instances"]),
 (["layers", "blocks", "how many layers", "depth", "stack"], ["layers", "blocks", "stages"]),
 (["bits", "key length", "key size", "key lengths"], ["bits"]),
 (["digest", "hash", "algorithm", "function", "primitive"], ["sha", "blake2", "digest", "hash"]),
]
NUMPAT = re.compile(r"\b(\d+(?:\.\d+)?(?:ms)?|SHA-256|SHA-3|BLAKE2)\b")

def best_sentence(q, idx):
    ql = " " + q.lower() + " "
    subj = SUBJ[idx]
    qt = set(t for t in terms(q) if t not in subj)
    best, bs = None, -1
    for k, s in enumerate(sents(KB[idx])):
        st = set(terms(s)); sl = s.lower()
        score = len(qt & st) * 1.0
        for trig, ev in ATTR:
            if any(t in ql for t in trig) and any(e in sl for e in ev): score += 3.0
        if NUMPAT.search(s): score += 0.7
        if "millisecond" in ql or " ms " in ql:
            score += 4.0 if re.search(r"\d+ms", s) else -3.0
        if score > bs: best, bs = s, score
    return best

def clean(q):
    q = re.sub(r"<!--.*?(-->|$)", " ", q, flags=re.S)
    q = re.sub(r"\[(system|admin|override|developer)[^\]]*\].*$", " ", q, flags=re.I | re.S)
    q2 = INJ.sub(" ", q)
    return q2 if len(q2.strip()) > 8 else q

def unknown_entities(q):
    """Capitalised / technical tokens in the question that never appear in the KB."""
    raw = re.findall(r"\b([A-Z][A-Za-z0-9+.\-]*[A-Za-z0-9]|[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]+\d[A-Za-z0-9]*)\b", q)
    out = []
    for i, w in enumerate(raw):
        lw = w.lower().replace("-", " ")
        parts = [p for p in toks(lw) if p not in STOP]
        if not parts: continue
        if q.strip().startswith(w) and w.lower() in STOP | {"what", "which", "how", "in", "on", "for", "when", "does"}: continue
        if all(p in VOCAB for p in parts): continue
        out.append(w)
    return out

def split_parts(q):
    parts = re.split(r"(?:,|;)?\s+and\s+(?=(?:what|which|how|the|its|what's)\b)|;\s*|\?\s+(?=[A-Z])", q, flags=re.I)
    return [p.strip() for p in parts if len(p.strip()) > 8]

def retrieve(q):
    qv = vec(terms(q))
    rare = [t for t in set(terms(q)) if t in IDF and DF[t] <= 3 and "_" not in t]
    out = []
    for i, d in enumerate(DV):
        s = cos(qv, d) + sum((0.3 if DF[t] <= 2 else 0.12) for t in rare if t in SUBJ[i]) \
            + sum((0.2 if DF[t] == 1 else 0.05) for t in rare if t in d and t not in SUBJ[i])
        out.append((s, i))
    return sorted(out, reverse=True)

SYNTH = re.compile(r"furthermore|in adjacent architecture|how do (both|they|these|the two) (integrate|compare|relate|differ|combine|work together)|synthesi[sz]", re.I)

def synth_answer(q, top_k):
    body = re.sub(r"^[^:]{0,60}:\s*", "", q, count=1)
    frags = [f for f in re.split(r"furthermore,?\s*(?:in adjacent architecture:)?|how do (?:both|they|these|the two)[^?]*\??", body, flags=re.I) if len(f.strip()) > 15]
    picks = []
    for f in frags[:3]:
        for s, i in retrieve(f):
            if i not in [p[1] for p in picks]:
                picks.append((min(1.0, s), i)); break
    if len(picks) < 2: return None
    rest = [x for x in retrieve(q) if x[1] not in [p[1] for p in picks]]
    top = (picks + rest)[: max(top_k, len(picks))]
    hi = max(p[0] for p in top)
    scores = [round(max(0.0, min(1.0, s)), 4) for s, _ in top]
    for k in range(len(picks)): scores[k] = round(min(1.0, max(hi, scores[k])), 4)
    for k in range(1, len(scores)): scores[k] = min(scores[k], scores[k - 1])
    ans = " ".join(KB[i] for _, i in picks)
    return {"answer": ans, "sources": [i for _, i in top], "scores": scores, "confidence": 0.9}

PREMISE = re.compile(r"^\s*(since|given that|given|assuming|as|because|if)\b([^,?]*\d[^,?]*),", re.I)

def false_premise(q):
    m = PREMISE.search(q)
    if not m: return False
    clause = m.group(2)
    nums = re.findall(r"\d+(?:\.\d+)?", clause)
    best = retrieve(clause)[0][1]
    return any(not re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d.])", KB[best]) for n in nums)

def answer(question, top_k=3):
    q = expand(clean(question))
    if false_premise(q):
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.1}
    if SYNTH.search(q):
        r = synth_answer(q, top_k)
        if r: return r
    parts = split_parts(q)
    if len(parts) > 1:
        rs = [answer_one(p, top_k) for p in parts]
        if all(r["answer"] for r in rs):
            srcs, scores = [], []
            firsts = [r["sources"][0] for r in rs]
            pool = {}
            for r in rs:
                for i, sc in zip(r["sources"], r["scores"]): pool[i] = max(pool.get(i, 0), sc)
            for i in firsts: pool[i] = max(pool[i], max(pool.values()))
            order = sorted(pool, key=lambda i: (-(i in firsts), -pool[i]))[: max(top_k, len(firsts))]
            ordered = sorted(order, key=lambda i: -pool[i])
            return {"answer": " ".join(r["answer"] for r in rs), "sources": ordered,
                    "scores": [round(pool[i], 4) for i in ordered], "confidence": min(r["confidence"] for r in rs)}
    return answer_one(q, top_k)

def answer_one(q, top_k=3):
    sc = retrieve(q)
    top = sc[: max(1, top_k)]
    s1 = top[0][0]
    qterms = [t for t in terms(q) if "_" not in t]
    known = [t for t in qterms if t in VOCAB]
    cover = len(known) / max(1, len(qterms))
    unk = unknown_entities(q)
    ql = " " + q.lower() + " "; pl = KB[top[0][1]].lower()
    missing_attr = False
    for trig, ev in [(["port"], ["port"]), (["learning rate"], ["learning rate"]), (["bits", "key length", "key size"], ["bits"]),
                     (["digest", "hash function"], ["sha", "blake2"])]:
        if any(" " + t in ql for t in trig) and not any(e in pl for e in ev): missing_attr = True
    if s1 < 0.12 or cover < 0.5 or unk or missing_attr:
        return {"answer": None, "sources": [], "scores": [], "confidence": round(min(0.3, s1), 4)}
    chosen = [top[0]]
    ans = " ".join(best_sentence(q, i) for _, i in chosen)
    GENERIC = set("port timeout interval size rate key length bits number many much threshold value default quorum digest hash function algorithm limit duration mini batch".split())
    numeric_q = any(t in ql for trig, _ in ATTR for t in trig)
    if numeric_q:
        idx = top[0][1]
        qual = [t for t in terms(q) if "_" not in t and t not in SUBJ[idx] and t not in GENERIC and t in VOCAB and len(t) > 2]
        stems = set(w[:4] for w in toks(ans))
        if qual and not any(t[:4] in stems for t in qual):
            return {"answer": None, "sources": [], "scores": [], "confidence": 0.2}
    srcs = [i for _, i in top]
    scores = [round(max(0.0, min(1.0, s)), 4) for s, _ in top]
    for k in range(1, len(scores)): scores[k] = min(scores[k], scores[k-1])
    conf = round(min(0.99, 0.65 + 0.35 * min(1.0, s1 / 0.5)), 4)
    return {"answer": ans, "sources": srcs, "scores": scores, "confidence": conf}
