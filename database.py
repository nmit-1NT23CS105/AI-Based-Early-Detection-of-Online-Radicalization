from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, desc, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from preprocessing import truncate_preview


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{(DATA_DIR / 'radicalization.db').as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    anonymized_text: Mapped[str] = mapped_column(Text)
    input_hash: Mapped[str] = mapped_column(String(64), index=True)
    language: Mapped[str] = mapped_column(String(24))
    predicted_label: Mapped[str] = mapped_column(String(24), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[int] = mapped_column(Integer, index=True)
    risk_level: Mapped[str] = mapped_column(String(24), index=True)
    emotion_label: Mapped[str] = mapped_column(String(24))
    anger_score: Mapped[float] = mapped_column(Float)
    fear_score: Mapped[float] = mapped_column(Float)
    hate_score: Mapped[float] = mapped_column(Float)
    probabilities_json: Mapped[str] = mapped_column(Text)
    explanation_json: Mapped[str] = mapped_column(Text)


class AlertRecord(Base):
    __tablename__ = "alert_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    severity: Mapped[str] = mapped_column(String(24), index=True)
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="active")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def _record_to_dict(record: AnalysisRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "created_at": record.created_at.isoformat(),
        "text_preview": truncate_preview(record.anonymized_text),
        "language": record.language,
        "predicted_label": record.predicted_label.replace("_", " ").title(),
        "confidence": round(record.confidence, 4),
        "risk_score": record.risk_score,
        "risk_level": record.risk_level,
        "emotion_label": record.emotion_label,
        "anger_score": round(record.anger_score, 2),
        "fear_score": round(record.fear_score, 2),
        "hate_score": round(record.hate_score, 2),
        "probabilities": json.loads(record.probabilities_json),
    }


def save_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    with SessionLocal() as session:
        record = AnalysisRecord(
            anonymized_text=result["anonymized_text"],
            input_hash=result["input_hash"],
            language=result["language"],
            predicted_label=result["prediction"]["label_key"],
            confidence=result["prediction"]["confidence"],
            risk_score=result["risk"]["score"],
            risk_level=result["risk"]["level"],
            emotion_label=result["emotion"]["label"],
            anger_score=result["emotion"]["scores"]["anger"],
            fear_score=result["emotion"]["scores"]["fear"],
            hate_score=result["emotion"]["scores"]["hate"],
            probabilities_json=json.dumps(result["prediction"]["probabilities"]),
            explanation_json=json.dumps(result["explanation"]["top_features"]),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        payload = _record_to_dict(record)
        if result["risk"]["level"] == "High Risk":
            save_alert(
                message=(
                    f"High-risk content detected for analysis #{record.id} "
                    f"with risk score {result['risk']['score']}."
                ),
                severity="high",
                session=session,
            )
        return payload


def save_alert(message: str, severity: str, session: Session | None = None) -> None:
    owns_session = session is None
    session = session or SessionLocal()
    try:
        alert = AlertRecord(severity=severity, message=message)
        session.add(alert)
        session.commit()
    finally:
        if owns_session:
            session.close()


def fetch_history(limit: int = 20) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        query = session.query(AnalysisRecord).order_by(desc(AnalysisRecord.created_at)).limit(limit)
        return [_record_to_dict(record) for record in query.all()]


def fetch_alerts(limit: int = 10) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        query = session.query(AlertRecord).order_by(desc(AlertRecord.created_at)).limit(limit)
        return [
            {
                "id": record.id,
                "created_at": record.created_at.isoformat(),
                "severity": record.severity,
                "message": record.message,
                "status": record.status,
            }
            for record in query.all()
        ]


def fetch_dashboard_summary() -> Dict[str, Any]:
    with SessionLocal() as session:
        total = session.query(func.count(AnalysisRecord.id)).scalar() or 0
        high_risk = (
            session.query(func.count(AnalysisRecord.id))
            .filter(AnalysisRecord.risk_level == "High Risk")
            .scalar()
            or 0
        )
        average_risk = round(session.query(func.avg(AnalysisRecord.risk_score)).scalar() or 0.0, 2)
        label_rows = (
            session.query(AnalysisRecord.predicted_label, func.count(AnalysisRecord.id))
            .group_by(AnalysisRecord.predicted_label)
            .all()
        )
        emotion_rows = (
            session.query(AnalysisRecord.emotion_label, func.count(AnalysisRecord.id))
            .group_by(AnalysisRecord.emotion_label)
            .all()
        )

        label_distribution = {label: count for label, count in label_rows}
        emotion_distribution = {label: count for label, count in emotion_rows}

        return {
            "total_analyses": total,
            "high_risk_count": high_risk,
            "average_risk": average_risk,
            "label_distribution": label_distribution,
            "emotion_distribution": emotion_distribution,
            "history": fetch_history(limit=12),
            "alerts": fetch_alerts(limit=8),
        }
