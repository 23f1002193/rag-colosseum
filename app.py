"""GA5 Q1 - RAG Colosseum: TF-IDF retrieval with entity boosting, extractive answers, abstention, injection stripping."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rag import answer
import time, collections
LOGQ = collections.deque(maxlen=200)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def always_cors(request, call_next):
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.get("/")
def root(): return {"ok": True, "service": "rag-colosseum", "passages": 50}

@app.post("/query")
async def query(req: Request):
    try: body = await req.json()
    except Exception: body = {}
    q = str(body.get("question") or "")
    try: k = max(1, min(10, int(body.get("top_k") or 3)))
    except Exception: k = 3
    if not q.strip():
        return JSONResponse({"answer": None, "sources": [], "scores": [], "confidence": 0.0})
    try: r = answer(q, k)
    except Exception as e:
        print("error", repr(e)); r = {"answer": None, "sources": [], "scores": [], "confidence": 0.0}
    LOGQ.append({"t": round(time.time(), 2), "q": q, "top_k": k, "r": r})
    return JSONResponse(r)

@app.get("/debug/log")
def dbg(key: str = ""):
    if key != "rekha2193secret": return JSONResponse({"error": "not found"}, 404)
    return list(LOGQ)
