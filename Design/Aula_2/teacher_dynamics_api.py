#!/usr/bin/env python3
from __future__ import annotations

"""

1o
    export OPENAI_API_KEY='...'
    python teacher_dynamics_api.py
Depois
    curl -X POST http://127.0.0.1:8000/teacher/dynamics \
      -H 'Content-Type: application/json' \
      -d '{"request":"Aula rapida de portugues sobre argumentacao para 9 ano"}'
Ou

python3 test_teacher_dynamics_local.py \
  --request "Dinamica para aula de ciencia para 6o ano com foco em engajamento" \
  --save-json
"""

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field


LESSONS_LIST_URL = "https://api.aprendizap.com.br/contents/lessons"
LESSON_PLANS_LIST_URL = "https://api.aprendizap.com.br/contents/lesson-plans"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CACHE_TTL_SECONDS = int(os.getenv("DYNAMICS_CACHE_TTL_SECONDS", "1800"))
MAX_HTTP_TIMEOUT = float(os.getenv("DYNAMICS_HTTP_TIMEOUT_SECONDS", "20"))

_STOPWORDS = {
    "a",
    "o",
    "e",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "um",
    "uma",
    "para",
    "com",
    "por",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "ao",
    "aos",
    "as",
    "os",
    "que",
    "como",
    "mais",
    "menos",
    "muito",
    "pouco",
    "sobre",
    "turma",
    "aula",
}

_SUBJECT_HINTS = {
    "portugues": ["portugues", "lingua portuguesa", "redacao", "leitura", "escrita", "gramatica"],
    "matematica": ["matematica", "algebra", "geometria", "equacao", "funcao", "estatistica"],
    "historia": ["historia", "idade media", "brasil colonia", "revolucao", "imperio"],
    "geografia": ["geografia", "territorio", "cartografia", "clima", "regiao"],
    "ciencias": ["ciencias", "biologia", "fisica", "quimica", "ecossistema"],
    "ingles": ["ingles", "english", "verb", "reading", "listening"],
    "arte": ["arte", "musica", "teatro", "danca", "artes visuais"],
    "educacao fisica": ["educacao fisica", "esporte", "lutas", "ginastica", "danca"],
}

_SUBJECT_API_NAMES = {
    "portugues": "Português",
    "matematica": "Matemática",
    "historia": "História",
    "geografia": "Geografia",
    "ciencias": "Ciências",
    "ingles": "Inglês",
    "arte": "Arte",
    "educacao fisica": "Educação Física",
}


@dataclass(frozen=True)
class SourceItem:
    source_id: str
    source_type: str
    title: str
    year: Optional[int]
    component: Optional[str]
    excerpt: str
    url: str
    score: float = 0.0


class DynamicsRequest(BaseModel):
    request: str = Field(..., min_length=8, description="Teacher free-text request. Minimal input expected.")
    duration_minutes: int = Field(default=50, ge=10, le=180)
    top_k_sources: int = Field(default=8, ge=4, le=20)
    model: str = Field(default=DEFAULT_MODEL)


class DynamicsResponse(BaseModel):
    plan_title: str
    objective: str
    inferred_context: Dict[str, Any]
    materials: List[Dict[str, Any]]
    steps: List[Dict[str, Any]]
    assessment: List[Dict[str, Any]]
    adaptations: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    verifier: Dict[str, Any]


app = FastAPI(title="Teacher Dynamics API", version="0.1.0")
_http = requests.Session()
_cache: Dict[str, Dict[str, Any]] = {}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    raw = _normalize(text).split()
    return [t for t in raw if len(t) >= 3 and t not in _STOPWORDS]


def _extract_year(text: str) -> Optional[int]:
    normalized = _normalize(text)
    patterns = [
        r"\b([6-9])\s*(?:o|º)?\s*ano\b",
        r"\b([1-3])\s*(?:o|º)?\s*ano\s*em\b",
        r"\b([1-3])\s*(?:o|º)?\s*em\b",
    ]
    for pat in patterns:
        match = re.search(pat, normalized)
        if match:
            return int(match.group(1))
    return None


def _extract_subject_key(text: str) -> Optional[str]:
    normalized = _normalize(text)
    for key, hints in _SUBJECT_HINTS.items():
        if any(h in normalized for h in hints):
            return key
    return None


def _infer_context(user_request: str, duration_minutes: int) -> Dict[str, Any]:
    subject_key = _extract_subject_key(user_request)
    year = _extract_year(user_request)
    constraints: List[str] = []

    normalized = _normalize(user_request)
    if any(x in normalized for x in ["rapida", "rapido", "curta", "pouco tempo", "corrida"]):
        constraints.append("tempo_reduzido")
    if any(x in normalized for x in ["heterogenea", "heterogeneo", "dificil", "defasagem", "mista"]):
        constraints.append("turma_heterogenea")
    if any(x in normalized for x in ["engajar", "engajamento", "desatenta", "indisciplina"]):
        constraints.append("engajamento")

    return {
        "subject_key": subject_key,
        "subject": _SUBJECT_API_NAMES.get(subject_key) if subject_key else None,
        "year": year,
        "duration_minutes": duration_minutes,
        "constraints": constraints,
    }


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        response = _http.get(url, params=params, timeout=MAX_HTTP_TIMEOUT)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Material API request failed: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Material API returned status={response.status_code} for {url}",
        )

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from material API: {url}") from exc


def _get_cache(key: str) -> Optional[Any]:
    data = _cache.get(key)
    if not data:
        return None
    if time.time() - data["ts"] > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return data["value"]


def _set_cache(key: str, value: Any) -> None:
    _cache[key] = {"ts": time.time(), "value": value}


def _paginate_docs(url: str, limit: int = 100, max_pages: int = 30) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    page = 1

    while page <= max_pages:
        payload = _http_get_json(url, params={"limit": limit, "page": page})
        page_docs = payload.get("docs") or []
        docs.extend(page_docs)

        has_next = bool(payload.get("hasNextPage"))
        if not has_next:
            break
        page += 1

    return docs


def _build_lesson_plan_sources() -> List[SourceItem]:
    cached = _get_cache("lesson_plans_sources")
    if cached is not None:
        return cached

    docs = _paginate_docs(LESSON_PLANS_LIST_URL, limit=100, max_pages=10)
    sources: List[SourceItem] = []

    for doc in docs:
        plan_id = str(doc.get("id"))
        if not plan_id:
            continue

        title = (doc.get("title") or "").strip()
        description = (doc.get("description") or "").strip()
        directions = (doc.get("directions") or "").strip()
        year = doc.get("year")

        cc = doc.get("curricularComponent") or {}
        component = cc.get("title") if isinstance(cc, dict) else None

        sections = doc.get("sections") or []
        section_texts: List[str] = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            sec_title = str(section.get("title") or "").strip()
            sec_content = str(section.get("content") or "").strip()
            if sec_title or sec_content:
                section_texts.append(f"{sec_title}: {sec_content}")

        text_parts = [title, description, directions, " | ".join(section_texts)]
        excerpt = "\n".join([t for t in text_parts if t]).strip()
        if not excerpt:
            continue

        sources.append(
            SourceItem(
                source_id=f"LP:{plan_id}",
                source_type="lesson_plan",
                title=title or f"Plano {plan_id}",
                year=year if isinstance(year, int) else None,
                component=component,
                excerpt=excerpt[:2400],
                url=f"https://api.aprendizap.com.br/contents/lesson-plans/{plan_id}",
            )
        )

    _set_cache("lesson_plans_sources", sources)
    return sources


def _build_lesson_sources() -> List[SourceItem]:
    cached = _get_cache("lesson_sources")
    if cached is not None:
        return cached

    docs = _paginate_docs(LESSONS_LIST_URL, limit=100, max_pages=30)
    sources: List[SourceItem] = []

    for unit in docs:
        unit_name = str(unit.get("name") or "").strip()
        lessons = unit.get("lessons") or []
        for lesson in lessons:
            if not isinstance(lesson, dict):
                continue
            lesson_id = str(lesson.get("id") or "")
            if not lesson_id:
                continue

            title = str(lesson.get("title") or "").strip()
            intro = str(lesson.get("intro") or "").strip()
            year = lesson.get("year") if isinstance(lesson.get("year"), int) else None

            comp = lesson.get("component") or {}
            component = comp.get("name") if isinstance(comp, dict) else None

            excerpt = f"{title}\nUnidade: {unit_name}\n{intro}".strip()
            if not excerpt:
                continue

            sources.append(
                SourceItem(
                    source_id=f"LS:{lesson_id}",
                    source_type="lesson",
                    title=title or f"Aula {lesson_id}",
                    year=year,
                    component=component,
                    excerpt=excerpt[:1800],
                    url=f"https://api.aprendizap.com.br/contents/lessons/{lesson_id}",
                )
            )

    _set_cache("lesson_sources", sources)
    return sources


def _lexical_score(query_tokens: Iterable[str], source: SourceItem, context: Dict[str, Any]) -> float:
    query_set = set(query_tokens)
    src_text = _normalize(f"{source.title} {source.excerpt} {source.component or ''}")
    src_tokens = set(_tokenize(src_text))
    if not src_tokens:
        return 0.0

    overlap = query_set & src_tokens
    score = float(len(overlap))

    if context.get("year") is not None:
        if source.year == context["year"]:
            score += 10.0
        elif source.year is not None:
            score -= 4.0

    expected_subject = context.get("subject")
    if expected_subject and source.component and _normalize(expected_subject) in _normalize(source.component):
        score += 8.0
    elif expected_subject and source.component:
        score -= 2.0

    if source.source_type == "lesson_plan":
        score += 1.0

    return score


def _prefilter_sources(all_sources: List[SourceItem], context: Dict[str, Any], top_k: int) -> List[SourceItem]:
    expected_subject = context.get("subject")
    expected_year = context.get("year")

    def subject_match(src: SourceItem) -> bool:
        if not expected_subject or not src.component:
            return False
        return _normalize(expected_subject) in _normalize(src.component)

    def year_match(src: SourceItem) -> bool:
        return expected_year is not None and src.year == expected_year

    subject_sources = [s for s in all_sources if subject_match(s)]
    year_sources = [s for s in all_sources if year_match(s)]
    both_sources = [s for s in all_sources if subject_match(s) and year_match(s)]

    # Prefer strong context matches when available, but keep fallback breadth.
    if len(both_sources) >= max(3, top_k // 2):
        ordered = both_sources + [s for s in subject_sources if s not in both_sources] + [s for s in year_sources if s not in both_sources and s not in subject_sources]
        return ordered
    if len(subject_sources) >= max(3, top_k // 2):
        return subject_sources + [s for s in all_sources if s not in subject_sources]
    if len(year_sources) >= max(3, top_k // 2):
        return year_sources + [s for s in all_sources if s not in year_sources]

    return all_sources


def _retrieve_sources(user_request: str, duration_minutes: int, top_k: int) -> Tuple[Dict[str, Any], List[SourceItem]]:
    context = _infer_context(user_request, duration_minutes)
    query_tokens = _tokenize(user_request)

    all_sources = _build_lesson_plan_sources() + _build_lesson_sources()
    if not all_sources:
        raise HTTPException(status_code=502, detail="No didactic sources retrieved from content APIs.")

    candidate_sources = _prefilter_sources(all_sources, context, top_k)

    ranked: List[SourceItem] = []
    for src in candidate_sources:
        score = _lexical_score(query_tokens, src, context)
        if score <= 0:
            continue
        ranked.append(
            SourceItem(
                source_id=src.source_id,
                source_type=src.source_type,
                title=src.title,
                year=src.year,
                component=src.component,
                excerpt=src.excerpt,
                url=src.url,
                score=score,
            )
        )

    if not ranked:
        ranked = [
            SourceItem(
                source_id=s.source_id,
                source_type=s.source_type,
                title=s.title,
                year=s.year,
                component=s.component,
                excerpt=s.excerpt,
                url=s.url,
                score=0.1,
            )
            for s in candidate_sources[: max(top_k, 8)]
        ]

    ranked.sort(key=lambda x: x.score, reverse=True)
    return context, ranked[:top_k]


def _extract_json_payload(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _load_api_key_from_env_file() -> Optional[str]:
    """
    Local fallback: read OPENAI_API_KEY from .env.
    Checked locations:
    1) current working directory
    2) same folder as this script
    """
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("OPENAI_API_KEY="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return value
        except OSError:
            continue
    return None


def _check_item_citations(items: Any, valid_ids: set[str], field_name: str, errors: List[str]) -> None:
    if not isinstance(items, list):
        errors.append(f"{field_name} must be a list")
        return

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"{field_name}[{idx}] must be an object")
            continue
        citations = item.get("citations")
        if not isinstance(citations, list) or len(citations) == 0:
            errors.append(f"{field_name}[{idx}] must include at least one citation")
            continue
        unknown = [c for c in citations if c not in valid_ids]
        if unknown:
            errors.append(f"{field_name}[{idx}] has invalid citation ids: {unknown}")


def _verify_plan(plan: Dict[str, Any], valid_ids: set[str]) -> Dict[str, Any]:
    errors: List[str] = []

    required_top = ["plan_title", "objective", "materials", "steps", "assessment", "adaptations"]
    for key in required_top:
        if key not in plan:
            errors.append(f"Missing key: {key}")

    _check_item_citations(plan.get("materials"), valid_ids, "materials", errors)
    _check_item_citations(plan.get("steps"), valid_ids, "steps", errors)
    _check_item_citations(plan.get("assessment"), valid_ids, "assessment", errors)
    _check_item_citations(plan.get("adaptations"), valid_ids, "adaptations", errors)

    if isinstance(plan.get("steps"), list):
        total_minutes = 0
        for idx, step in enumerate(plan["steps"]):
            minutes = step.get("minutes") if isinstance(step, dict) else None
            if not isinstance(minutes, int) or minutes <= 0:
                errors.append(f"steps[{idx}].minutes must be a positive integer")
            else:
                total_minutes += minutes
        if total_minutes <= 0:
            errors.append("Total step duration must be > 0")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "all_claims_cited": len(errors) == 0,
        "unsupported_claims_count": len(errors),
    }


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = _load_api_key_from_env_file()
        if api_key:
            # Keep process env in sync after local fallback.
            os.environ["OPENAI_API_KEY"] = api_key
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "OPENAI_API_KEY is not configured for this server process.",
                "how_to_fix": [
                    "Start server in the same shell where OPENAI_API_KEY is exported.",
                    "Or create .env with OPENAI_API_KEY=sk-... in this project folder.",
                ],
                "debug": {
                    "cwd": str(Path.cwd()),
                    "pid": os.getpid(),
                    "python": os.sys.executable,
                    "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
                },
            },
        )
    return OpenAI(api_key=api_key)


def _build_messages(
    request: DynamicsRequest,
    inferred_context: Dict[str, Any],
    sources: List[SourceItem],
    previous_plan: Optional[Dict[str, Any]] = None,
    repair_errors: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    compact_sources = []
    for src in sources:
        compact_sources.append(
            {
                "source_id": src.source_id,
                "source_type": src.source_type,
                "title": src.title,
                "component": src.component,
                "year": src.year,
                "excerpt": src.excerpt,
                "url": src.url,
            }
        )

    system = (
        "You are a pedagogical planning assistant. "
        "Use ONLY the provided sources. "
        "Return valid JSON only. "
        "Every item in materials, steps, assessment, and adaptations must include citations with source_id values. "
        "Do not cite sources outside the provided source ids. "
        "Language: Portuguese (pt-BR)."
    )

    user_payload: Dict[str, Any] = {
        "task": "Generate a classroom dynamic with minimal teacher input.",
        "teacher_request": request.request,
        "duration_minutes": request.duration_minutes,
        "inferred_context": inferred_context,
        "output_schema": {
            "plan_title": "string",
            "objective": "string",
            "materials": [{"text": "string", "citations": ["source_id"]}],
            "steps": [
                {"title": "string", "minutes": "int", "activity": "string", "citations": ["source_id"]}
            ],
            "assessment": [{"text": "string", "citations": ["source_id"]}],
            "adaptations": [{"text": "string", "citations": ["source_id"]}],
        },
        "sources": compact_sources,
    }

    if previous_plan is not None and repair_errors is not None:
        user_payload["repair_mode"] = {
            "previous_plan": previous_plan,
            "errors": repair_errors,
            "instruction": "Fix the plan so all validation errors are removed while keeping citations valid.",
        }

    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)}]


def _generate_with_openai(request: DynamicsRequest, context: Dict[str, Any], sources: List[SourceItem]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    client = _openai_client()
    valid_ids = {s.source_id for s in sources}

    messages = _build_messages(request, context, sources)
    try:
        completion = client.chat.completions.create(
            model=request.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI generation failed: {exc}") from exc

    raw_text = completion.choices[0].message.content or "{}"
    try:
        plan = _extract_json_payload(raw_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI returned non-JSON output: {exc}") from exc

    verifier = _verify_plan(plan, valid_ids)
    if verifier["is_valid"]:
        return plan, verifier

    # One repair attempt with explicit validation errors.
    repair_messages = _build_messages(request, context, sources, previous_plan=plan, repair_errors=verifier["errors"])
    try:
        repair_completion = client.chat.completions.create(
            model=request.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=repair_messages,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI repair call failed: {exc}") from exc

    repair_text = repair_completion.choices[0].message.content or "{}"
    try:
        repaired_plan = _extract_json_payload(repair_text)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Repair output not parseable as JSON: {exc}") from exc

    repaired_verifier = _verify_plan(repaired_plan, valid_ids)
    if not repaired_verifier["is_valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Plan blocked: not all claims are fully cited with valid source ids.",
                "errors": repaired_verifier["errors"],
            },
        )

    return repaired_plan, repaired_verifier


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "teacher_dynamics_api"}


@app.get("/debug/env")
def debug_env() -> Dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY")
    masked_prefix = None
    if key:
        masked_prefix = key[:7] + "..."
    return {
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
        "python": os.sys.executable,
        "openai_api_key_present": bool(key),
        "openai_api_key_prefix": masked_prefix,
        "env_file_detected": (Path.cwd() / ".env").exists() or (Path(__file__).resolve().parent / ".env").exists(),
    }


@app.post("/teacher/dynamics", response_model=DynamicsResponse)
def teacher_dynamics(payload: DynamicsRequest) -> Dict[str, Any]:
    context, sources = _retrieve_sources(
        user_request=payload.request,
        duration_minutes=payload.duration_minutes,
        top_k=payload.top_k_sources,
    )

    plan, verifier = _generate_with_openai(payload, context, sources)

    citations = [
        {
            "source_id": s.source_id,
            "source_type": s.source_type,
            "title": s.title,
            "component": s.component,
            "year": s.year,
            "url": s.url,
            "excerpt": s.excerpt,
            "relevance_score": round(s.score, 3),
        }
        for s in sources
    ]

    return {
        "plan_title": plan.get("plan_title", "Dinamica de aula"),
        "objective": plan.get("objective", ""),
        "inferred_context": context,
        "materials": plan.get("materials", []),
        "steps": plan.get("steps", []),
        "assessment": plan.get("assessment", []),
        "adaptations": plan.get("adaptations", []),
        "citations": citations,
        "verifier": verifier,
    }


if __name__ == "__main__":
    uvicorn.run(
        "teacher_dynamics_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
