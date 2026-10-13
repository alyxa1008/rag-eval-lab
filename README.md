# rag-eval-lab

한국어 ESG 보고서(PDF)를 대상으로 하이브리드 검색(벡터 + BM25)의 품질을 **숫자로** 평가하는 프로젝트입니다.

## 문제 정의

이전에 대학 상담 RAG 챗봇을 만들어 운영하면서 세 가지 한계가 있었습니다.

- 튜닝과 검증에 같은 질의 세트를 써서 held-out이 없었다.
- 벡터 60% + BM25 40% 비율을 다른 비율과 비교하지 않고 채택했다.
- 검색 품질을 Recall@k 같은 지표로 재지 않았다.

이 저장소는 그 세 가지를 직접 해보고, "held-out 세트에서 Recall@5가 몇 %였다"를 말할 수 있게 만드는 것이 목표입니다.

## 파이프라인

| 단계 | 방법 |
|---|---|
| 파싱 | PyMuPDF(본문) + pdfplumber(테두리 있는 표) + 단어 좌표 기반 재구성(테두리 없는 숫자 표). 표는 행 단위로 `항목 \| 단위 \| 2022년: x \| 2023년: y` 형태로 텍스트화하고 N행마다 분할 |
| 청킹 | 페이지 기준 텍스트 청크 + 표 청크 분리. 모든 청크에 `page` 메타데이터 |
| 임베딩 | BAAI/bge-m3 로컬 실행 (sentence-transformers), 결과 SQLite 캐시. 설정으로 OpenAI text-embedding-3-small 전환 가능 |
| 벡터 DB | ChromaDB (로컬) |
| 키워드 검색 | rank_bm25 + kiwipiepy 형태소 분석 |
| 융합 | min-max 정규화 후 `w*vector + (1-w)*bm25` |
| 리랭커 (선택) | BAAI/bge-reranker-v2-m3 |

## 평가셋

- `data/eval_set.csv`, 40문항(number 15 / table 8 / text 12 / multi 5). 질문은 문서 표현을 그대로 베끼지 않고 바꿔 썼고, 정답과 `gold_pages`는 원문 쪽과 대조해 검수했습니다.
- 컬럼: `id, question, answer, gold_pages, type, split`
- 40문항 중 10개(25%)를 시드 고정 무작위 추출로 `heldout`에 배정했고, 튜닝 중에는 보지 않습니다. heldout은 `--split heldout`을 명시해야만 실행됩니다.

## 지표

- Recall@k (k=1,3,5): 상위 k개 청크 중 `gold_pages`에 속한 청크가 하나라도 있으면 성공
- MRR: 처음 맞춘 청크의 순위 역수 평균
- type별(number / table / text / multi) 분해

## 실행

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env            # OpenAI 임베딩을 쓸 때만 OPENAI_API_KEY 입력 (기본은 로컬 bge-m3, 키 불필요)

# M0: 파싱 결과 확인 (페이지별 글자 수·표 수·표 행 수)
python -m rag_eval parse --config configs/base.yaml
python -m rag_eval parse --config configs/base.yaml --page 12   # 특정 쪽 청크 전문 보기

# M1: 인덱싱(파싱 → 임베딩(캐시) → ChromaDB) 후 단건 검색
python -m rag_eval index --config configs/base.yaml
python -m rag_eval search --config configs/base.yaml -q "2024년 Scope 1 온실가스 배출량"
python -m rag_eval search --config configs/base.yaml -q "임직원 수" --mode bm25   # API 키 없이 BM25만
python -m rag_eval search --config configs/base.yaml -q "임직원 수" -w 0.3        # 벡터 가중치 지정

# 평가셋 형식 검사
python -m rag_eval check-eval-set

# M2: 평가 (기본은 dev 세트만. heldout은 설정 확정 후 마지막에 딱 한 번)
python -m rag_eval run --config configs/base.yaml
python -m rag_eval run --config configs/base.yaml -w 0.3          # 가중치만 바꿔서 실행
python -m rag_eval run --config configs/base.yaml --split heldout  # 최종 1회
# -> results/<config>/<split>/<run>/ 에 per_query.csv, summary.csv, summary.md, run_info.json

pytest
```

## 문서

원본 PDF는 저작권 때문에 커밋하지 않습니다. 아래 출처에서 내려받아 `data/raw/`에 두세요.

- (문서 선정 후 기재)

## 결과

(실험 후 실제 실행 결과만 기재)

## 실패 분석

(실험 후 기재)

## 효과가 없었던 것

(실험 후 기재)

## 한계

- 76~79쪽(제3자 인증·검증 의견서)은 스캔 이미지라 텍스트가 추출되지 않는다. OCR은 범위 밖.
- 연도 × 사업부문처럼 헤더가 2단인 표(72~74쪽)는 숫자는 추출되지만 열 이름이 연도까지만 붙는다.
- 본문 텍스트 청크에도 표 셀이 평문으로 남아 있어 표 청크와 내용이 일부 겹친다.
- (실험 후 추가)
