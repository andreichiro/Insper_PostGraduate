# Teacher Dynamics API (MVP)

Endpoint local para gerar dinâmica de aula com entrada mínima (`request`) e validação de citação.

## O que este MVP faz
- Professor envia apenas texto livre (`request`).
- API infere contexto (disciplina/ano/restrições).
- API busca material didático oficial (`lessons` + `lesson-plans`).
- API chama OpenAI para gerar a dinâmica.
- API bloqueia saída sem citação válida.

## 1) Quick start (mínimo obrigatório)
Se você já tem `fastapi`, `uvicorn`, `requests` e `openai` instalados:

```bash
cd "/Users/akatsurada/Documents/INSPER/Design/Aula_2"

# importante: use chave real, não "..."
OPENAI_API_KEY="sk-..." python3 teacher_dynamics_api.py
```

Em outro terminal:

```bash
cd "/Users/akatsurada/Documents/INSPER/Design/Aula_2"
python3 test_teacher_dynamics_local.py \
  --request "Aula rapida de portugues sobre argumentacao para 9 ano" \
  --save-json
```

## 2) Checar dependências (opcional)

```bash
python3 - <<'PY'
import fastapi, uvicorn, requests, openai
print("deps ok")
PY
```

Se falhar, instale dependências (passo 3).

## 3) Instalar dependências (se necessário)

```bash
cd "/Users/akatsurada/Documents/INSPER/Design/Aula_2"

# opcional
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install fastapi uvicorn requests openai
```

## 4) Configurar chave OpenAI

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"  # opcional
```

Valide na mesma shell onde vai subir a API:

```bash
python3 - <<'PY'
import os
k=os.getenv('OPENAI_API_KEY')
print('OPENAI_API_KEY_set =', bool(k))
print('OPENAI_API_KEY_prefix =', (k[:7]+'...') if k else None)
PY
```

## 5) Subir API local

```bash
python3 teacher_dynamics_api.py
```

A API sobe em `http://127.0.0.1:8000`.

## 6) Diagnóstico rápido de ambiente (importante)
Se der erro de chave, cheque o que o **processo do servidor** está vendo:

```bash
curl -s "http://127.0.0.1:8000/debug/env"
```

Campos esperados:
- `openai_api_key_present: true`
- `openai_api_key_prefix: sk-...`

## 7) Testar sua request (forma mais rápida)

```bash
python3 test_teacher_dynamics_local.py \
  --request "Aula rapida de portugues sobre argumentacao para 9 ano" \
  --save-json
```

## 8) Testar requests diferentes

```bash
python3 test_teacher_dynamics_local.py --request "Aula curta de matematica sobre funcoes para 9 ano"
python3 test_teacher_dynamics_local.py --request "Turma heterogenea, preciso trabalhar leitura critica em portugues 8 ano"
python3 test_teacher_dynamics_local.py --request "Aula de historia 7 ano sobre idade media com foco em engajamento"
python3 test_teacher_dynamics_local.py --request "Preciso de dinamica rapida de ciencias para 6 ano com avaliacao formativa"
```

Modo interativo:

```bash
python3 test_teacher_dynamics_local.py --interactive --save-json
```

## 9) Teste via cURL

```bash
curl -X POST "http://127.0.0.1:8000/teacher/dynamics" \
  -H "Content-Type: application/json" \
  -d '{
    "request": "Aula rapida de portugues sobre argumentacao para 9 ano",
    "duration_minutes": 45,
    "top_k_sources": 8,
    "model": "gpt-4o-mini"
  }'
```

## 10) Onde olhar resultado
- Terminal do teste: resumo (contexto, steps, verificador, fontes).
- JSON completo (`--save-json`):
  - `/Users/akatsurada/Documents/INSPER/Design/Aula_2/analysis_output/tests/teacher_dynamics_response_*.json`

## 11) Como validar citação aprovada
No JSON de resposta:
- `verifier.is_valid == true`
- `verifier.all_claims_cited == true`
- `verifier.unsupported_claims_count == 0`

Se falhar, a API tenta 1 reparo automático; se continuar inválido, retorna `422`.

## 12) Arquivos
- API: `/Users/akatsurada/Documents/INSPER/Design/Aula_2/teacher_dynamics_api.py`
- Tester: `/Users/akatsurada/Documents/INSPER/Design/Aula_2/test_teacher_dynamics_local.py`
- Guia: `/Users/akatsurada/Documents/INSPER/Design/Aula_2/teacher_dynamics_api_README.md`
