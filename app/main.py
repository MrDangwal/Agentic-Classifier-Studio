import csv
import io
import json
import os
import time
from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from joblib import dump, load
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "app.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SECRET_KEY = os.environ.get("APP_SECRET", "dev-secret-change")
serializer = URLSafeSerializer(SECRET_KEY, salt="session")

engine = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    rows: Mapped[List["Row"]] = relationship(back_populates="dataset")
    agents: Mapped[List["Agent"]] = relationship(back_populates="dataset")

class Row(Base):
    __tablename__ = "rows"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    text: Mapped[str] = mapped_column(Text)
    classification: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    predicted_label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dataset: Mapped[Dataset] = relationship(back_populates="rows")

class Agent(Base):
    __tablename__ = "agents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    name: Mapped[str] = mapped_column(String(255))
    model_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    labels_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    dataset: Mapped[Dataset] = relationship(back_populates="agents")

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

ADMIN_USER = "admin"
ADMIN_PASS = "1234"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
LLM_ENABLED = bool(os.environ.get("OPENAI_API_KEY"))

POSITIVE_KEYWORDS = {
    "love", "loved", "great", "amazing", "fantastic", "excellent", "happy", "good", "perfect",
    "quick", "fast", "helpful", "recommend", "best", "nice", "satisfied",
}
NEGATIVE_KEYWORDS = {
    "hate", "hated", "bad", "terrible", "awful", "worst", "broken", "poor", "slow", "late",
    "refund", "expensive", "disappointed", "useless", "confusing", "never",
}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_login(request: Request):
    cookie = request.cookies.get("session")
    if not cookie:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        data = serializer.loads(cookie)
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid session")
    if data.get("user") != ADMIN_USER:
        raise HTTPException(status_code=401, detail="Invalid user")
    return data

class BulkLabelItem(BaseModel):
    row_id: int
    classification: str

class BulkLabelRequest(BaseModel):
    dataset_id: int
    items: List[BulkLabelItem]

class AgentCreateRequest(BaseModel):
    dataset_id: int
    name: str

class TrainRequest(BaseModel):
    label_budget: int = 50
    target_f1: float = 0.85

def keyword_agent(text: str) -> tuple[Optional[str], float]:
    tokens = {t.strip(".,!?;:()[]\"'").lower() for t in text.split()}
    pos = len(tokens & POSITIVE_KEYWORDS)
    neg = len(tokens & NEGATIVE_KEYWORDS)
    if pos == 0 and neg == 0:
        return None, 0.33
    if pos == neg:
        return None, 0.45
    label = "positive" if pos > neg else "negative"
    confidence = min(0.9, 0.55 + (abs(pos - neg) / (pos + neg)))
    return label, confidence

def self_check_agent(text: str) -> float:
    # Penalize ambiguous or negated text to simulate a reviewer agent.
    flags = ["not", "never", "no", "unsure", "maybe", "confusing"]
    return -0.12 if any(f in text.lower() for f in flags) else 0.0

def llm_agent(text: str, labels: List[str]) -> tuple[Optional[str], float]:
    if not LLM_ENABLED:
        return None, 0.0
    label_list = labels or ["positive", "negative", "neutral"]
    system = (
        "You are a strict text classifier. "
        "Return JSON with keys: label, confidence (0-1). "
        f"Allowed labels: {', '.join(label_list)}."
    )
    prompt = f"Text: {text}"
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        data = json.loads(result.content)
        label = str(data.get("label", "")).strip()
        conf = float(data.get("confidence", 0.5))
        if label not in label_list:
            return None, 0.0
        return label, max(0.0, min(0.99, conf))
    except Exception:
        return None, 0.0

def orchestrate_prediction(
    text: str,
    model_label: Optional[str],
    model_conf: Optional[float],
    llm_label: Optional[str],
    llm_conf: Optional[float],
) -> tuple[str, float]:
    keyword_label, keyword_conf = keyword_agent(text)
    adjust = self_check_agent(text)

    if model_label:
        final_label = model_label
        final_conf = float(model_conf or 0.5)
        if llm_label and llm_label == model_label:
            final_conf = min(0.99, final_conf + 0.15)
        elif llm_label and llm_label != model_label:
            final_conf = max(0.3, final_conf - 0.2)
        if keyword_label and keyword_label == model_label:
            final_conf = min(0.99, final_conf + 0.1)
    else:
        if llm_label:
            final_label = llm_label
            final_conf = float(llm_conf or 0.5)
        else:
            final_label = keyword_label or "unknown"
            final_conf = keyword_conf

    final_conf = max(0.0, min(0.99, final_conf + adjust))
    return final_label, final_conf

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    cookie = request.cookies.get("session")
    if not cookie:
        return templates.TemplateResponse("login.html", {"request": request})
    try:
        serializer.loads(cookie)
    except BadSignature:
        return templates.TemplateResponse("login.html", {"request": request})
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if username != ADMIN_USER or password != ADMIN_PASS:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = serializer.dumps({"user": username, "ts": int(time.time())})
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie("session", token, httponly=True)
    return response

@app.post("/logout")
def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("session")
    return response

@app.post("/datasets/upload")
def upload_dataset(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    require_login(request)
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV allowed")
    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames != ["text", "classification"]:
        raise HTTPException(status_code=400, detail="CSV must have columns: text, classification")
    dataset = Dataset(name=file.filename)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    rows = []
    for row in reader:
        text = (row.get("text") or "").strip()
        classification = (row.get("classification") or "").strip() or None
        rows.append(Row(dataset_id=dataset.id, text=text, classification=classification))
    db.bulk_save_objects(rows)
    db.commit()
    return {"dataset_id": dataset.id, "rows": len(rows)}

@app.get("/datasets")
def list_datasets(request: Request, db: Session = Depends(get_db)):
    require_login(request)
    datasets = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
    return [
        {"id": d.id, "name": d.name, "created_at": d.created_at.isoformat()}
        for d in datasets
    ]

@app.get("/datasets/{dataset_id}/rows")
def get_rows(
    request: Request,
    dataset_id: int,
    filter: str = "all",
    limit: int = 50,
    db: Session = Depends(get_db),
):
    require_login(request)
    stmt = select(Row).where(Row.dataset_id == dataset_id)
    if filter == "unlabeled":
        stmt = stmt.where(Row.classification.is_(None))
    elif filter == "uncertain":
        stmt = stmt.where(Row.classification.is_(None)).where(Row.confidence.is_not(None)).order_by(Row.confidence.asc())
    rows = db.scalars(stmt.limit(limit)).all()
    return [
        {
            "id": r.id,
            "text": r.text,
            "classification": r.classification,
            "predicted_label": r.predicted_label,
            "confidence": r.confidence,
        }
        for r in rows
    ]

@app.post("/labels/bulk")
def bulk_label(request: Request, payload: BulkLabelRequest, db: Session = Depends(get_db)):
    require_login(request)
    row_ids = [i.row_id for i in payload.items]
    rows = db.scalars(select(Row).where(Row.dataset_id == payload.dataset_id, Row.id.in_(row_ids))).all()
    by_id = {r.id: r for r in rows}
    for item in payload.items:
        if item.row_id in by_id:
            by_id[item.row_id].classification = item.classification
    db.commit()
    return {"updated": len(payload.items)}

@app.post("/agents")
def create_agent(request: Request, payload: AgentCreateRequest, db: Session = Depends(get_db)):
    require_login(request)
    agent = Agent(dataset_id=payload.dataset_id, name=payload.name)
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return {"agent_id": agent.id}

@app.get("/agents/{agent_id}/metrics")
def get_metrics(request: Request, agent_id: int, db: Session = Depends(get_db)):
    require_login(request)
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return json.loads(agent.metrics_json) if agent.metrics_json else {}

@app.post("/agents/{agent_id}/train")
def train_agent(request: Request, agent_id: int, payload: TrainRequest, db: Session = Depends(get_db)):
    require_login(request)
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    rows = db.scalars(select(Row).where(Row.dataset_id == agent.dataset_id, Row.classification.is_not(None))).all()
    if len(rows) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 labeled rows")

    texts = [r.text for r in rows]
    labels = [r.classification for r in rows]
    unique_labels = sorted(set(labels))

    # train/valid split (simple)
    split = max(1, int(0.8 * len(rows)))
    train_texts, valid_texts = texts[:split], texts[split:]
    train_labels, valid_labels = labels[:split], labels[split:]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    metrics = {}
    if valid_texts:
        X_valid = vectorizer.transform(valid_texts)
        preds = model.predict(X_valid)
        metrics["accuracy"] = float(accuracy_score(valid_labels, preds))
        metrics["macro_f1"] = float(f1_score(valid_labels, preds, average="macro"))
        metrics["confusion_matrix"] = confusion_matrix(valid_labels, preds, labels=unique_labels).tolist()
        metrics["labels"] = unique_labels

    model_path = os.path.join(MODEL_DIR, f"agent_{agent.id}.joblib")
    dump({"vectorizer": vectorizer, "model": model}, model_path)

    agent.model_path = model_path
    agent.labels_json = json.dumps(unique_labels)
    agent.metrics_json = json.dumps(metrics)
    db.commit()

    return {"agent_id": agent.id, "metrics": metrics}

@app.post("/agents/{agent_id}/run")
def run_agent(request: Request, agent_id: int, dataset_id: int, db: Session = Depends(get_db)):
    require_login(request)
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if dataset_id != agent.dataset_id:
        raise HTTPException(status_code=400, detail="Agent and dataset mismatch")

    rows = db.scalars(select(Row).where(Row.dataset_id == dataset_id, Row.classification.is_(None))).all()
    if not rows:
        return {"updated": 0}

    labels = []
    preds = []
    probs = []
    if agent.model_path and os.path.exists(agent.model_path):
        bundle = load(agent.model_path)
        vectorizer = bundle["vectorizer"]
        model = bundle["model"]
        X = vectorizer.transform([r.text for r in rows])
        probs = model.predict_proba(X)
        labels = model.classes_
        preds = probs.argmax(axis=1)

    updated = 0
    for i, row in enumerate(rows):
        model_label = None
        model_conf = None
        if labels != []:
            model_label = str(labels[preds[i]])
            sorted_probs = sorted(probs[i], reverse=True)
            model_conf = float(sorted_probs[0]) if sorted_probs else 0.0
        llm_label, llm_conf = llm_agent(row.text, json.loads(agent.labels_json) if agent.labels_json else [])
        final_label, final_conf = orchestrate_prediction(
            row.text, model_label, model_conf, llm_label, llm_conf
        )
        row.predicted_label = final_label
        row.confidence = final_conf
        updated += 1
    db.commit()
    return {"updated": updated}

@app.post("/agents/{agent_id}/export")
def export_agent(request: Request, agent_id: int, db: Session = Depends(get_db)):
    require_login(request)
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    bundle = {
        "agent_id": agent.id,
        "dataset_id": agent.dataset_id,
        "name": agent.name,
        "labels": json.loads(agent.labels_json) if agent.labels_json else [],
        "metrics": json.loads(agent.metrics_json) if agent.metrics_json else {},
    }
    return JSONResponse(content=bundle)

@app.post("/agents/import")
def import_agent(request: Request, payload: dict, db: Session = Depends(get_db)):
    require_login(request)
    name = payload.get("name", "imported-agent")
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id required")
    agent = Agent(dataset_id=dataset_id, name=name, labels_json=json.dumps(payload.get("labels", [])), metrics_json=json.dumps(payload.get("metrics", {})))
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return {"agent_id": agent.id}
