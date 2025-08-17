# main_final.py  (주소 기반 파일명 / biz_id 미포함)
import argparse
from pathlib import Path
import pandas as pd
import re

from config import (
    PATH_BUSINESS, PATH_REVIEW,
    PATH_REVIEWS_FILTERED, PATH_SENT_WITH_SENT, PATH_WITH_TOPICS,
    SENTIMENT_MODEL, EMBEDDING_MODEL
)
from modules.find_business_ids import find_business_ids
from modules.filter_reviews import filter_reviews_by_business_ids
from modules.sentence_sentiment import run_sentence_sentiment
from modules.business_meta import load_business_meta
from modules.topic_model_final import apply_bertopic_for_business


def parse_args():
    p = argparse.ArgumentParser(
        description="Mini Yelp pipeline: N businesses → sentence-level sentiment → BERTopic (per-store / pooled)"
    )
    # 대상 비즈니스 선택
    p.add_argument("--name-substr", type=str, default=None)
    p.add_argument("--category", type=str, default="Restaurants")
    p.add_argument("--city", type=str, default=None)
    p.add_argument("--state", type=str, default=None)
    p.add_argument("--biz-id", action="append", help="Explicit business_id (can repeat)")
    p.add_argument("--limit", type=int, default=2)

    # 토픽 실행 스위치
    p.add_argument("--do-topic", action="store_true")

    # 토픽 스코프
    p.add_argument("--topic-scope", type=str, default="per-store",
                   choices=["per-store", "pooled", "both"],
                   help="Run BERTopic per store, pooled across stores, or both")

    # 임계치(점포별 최소 문장/리뷰 수)
    p.add_argument("--per-store-min-sentences", type=int, default=80,
                   help="Run per-store topic modeling only if sentence count ≥ this (default: 80)")
    p.add_argument("--per-store-min-reviews", type=int, default=0,
                   help="Optional: pre-filter stores by raw review count (0=ignore)")

    # 토픽 파라미터
    p.add_argument("--min-topic-size", type=int, default=None)
    p.add_argument("--max-topics", dest="nr_topics", type=int, default=None)

    # 제로샷 라벨링
    p.add_argument("--zeroshot", action="store_true")
    p.add_argument("--zeroshot-online", action="store_true")
    p.add_argument("--zeroshot-min-prob", type=float, default=0.4)
    p.add_argument("--label-mode", type=str, default="auto", choices=["core","auto","full"])

    # 출력 경로/파일명  ← 기본 템플릿에서 biz_id 제거, addr_slug만 사용
    p.add_argument("--out-dir", type=str, default=None,
                   help="Directory for topic outputs; default = parent of PATH_WITH_TOPICS")
    p.add_argument("--filename-template", type=str,
                   default="with_topics__{scope}__{addr_slug}.csv",
                   help="Placeholders: {scope}, {addr_slug}  (biz_id는 사용하지 않음)")

    # 실행 요약 인덱스 파일
    p.add_argument("--write-index", action="store_true",
                   help="Write an index CSV summarizing per-store run status")

    return p.parse_args()


def slugify(s: str, max_len: int = 60) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "store"


def main():
    args = parse_args()

    # 1) 대상 business_id 결정
    if args.biz_id:
        biz_ids = args.biz_id
    else:
        biz_ids = find_business_ids(
            business_json_path=str(PATH_BUSINESS),
            name_substring=args.name_substr,
            category_keyword=args.category,
            city=args.city,
            state=args.state,
            limit=args.limit
        )
    if not biz_ids:
        raise SystemExit("No business matched. Try different filters or supply --biz-id.")
    print("[Picked business_id(s)]", biz_ids)

    # 1.5) business.jsonl에서 주소/이름 메타 로드
    # load_business_meta()는 {bid: {"name":..., "address":...}} 형태 반환
    id2meta = load_business_meta(str(PATH_BUSINESS), biz_ids)

    # 주소 → 슬러그 매핑(+충돌 회피)
    seen_addr = {}
    def addr_slug_of(bid: str) -> str:
        meta = id2meta.get(bid, {}) or {}
        base = meta.get("address") or meta.get("name") or "store"
        s = slugify(base, max_len=60)
        # 중복 주소 처리: -2, -3 ... 접미사
        cnt = seen_addr.get(s, 0) + 1
        seen_addr[s] = cnt
        return s if cnt == 1 else f"{s}-{cnt}"

    # 2) 리뷰 필터 → 통합 JSONL
    kept = filter_reviews_by_business_ids(
        review_json_path=str(PATH_REVIEW),
        target_business_ids=biz_ids,
        out_jsonl_path=str(PATH_REVIEWS_FILTERED),
    )
    print(f"[Filtered reviews] kept={kept} → {PATH_REVIEWS_FILTERED}")

    # 3) 문장 단위 감성
    run_sentence_sentiment(
        filtered_jsonl_path=str(PATH_REVIEWS_FILTERED),
        out_csv_path=str(PATH_SENT_WITH_SENT),
        sentiment_model=SENTIMENT_MODEL,
        max_chars=256,
        batch_size=64
    )
    print(f"[Sentence sentiment] → {PATH_SENT_WITH_SENT}")

    # 3.5) 점포별 문장/리뷰 수 집계 (임계치 필터용)
    sent_df = pd.read_csv(PATH_SENT_WITH_SENT)
    sent_counts = sent_df.groupby("business_id").size().to_dict()

    # 리뷰 수 임계치(선택)
    review_counts = None
    if args.per_store_min_reviews > 0:
        if "review_id" in sent_df.columns:
            review_counts = (
                sent_df.drop_duplicates(["business_id", "review_id"])
                      .groupby("business_id").size().to_dict()
            )
        else:
            print("[Warn] per-store-min-reviews requested but review_id not in sentence CSV; skipping this filter.")

    # 4) 출력 디렉토리
    out_dir = Path(args.out_dir) if args.out_dir else Path(PATH_WITH_TOPICS).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 공통 러너
    def run_topic(scope: str, business_id, out_base_path: Path):
        model = apply_bertopic_for_business(
            sentences_csv_path=str(PATH_SENT_WITH_SENT),
            out_csv_path=str(out_base_path),
            business_id=business_id,                # None/"__ALL__" → 통합
            embedding_model=EMBEDDING_MODEL,
            min_topic_size=(None if args.min_topic_size in (None, 0) else args.min_topic_size),
            nr_topics=args.nr_topics,
            enable_zeroshot=bool(args.zeroshot),
            zeroshot_online=bool(args.zeroshot_online),
            zeroshot_min_prob=float(args.zeroshot_min_prob),
            label_mode=args.label_mode,
        )
        base = str(out_base_path)
        print(f"[BERTopic done] scope={scope} business_id={business_id} → {base}")
        print("Artifacts:")
        print(" - per-sentence topics:", base)
        print(" - topic summary      :", base.replace('.csv', '_topic_summary.csv'))
        print(" - topic trend monthly:", base.replace('.csv', '_topic_trend_by_month.csv'))
        print(" - topic examples     :", base.replace('.csv', '_topic_examples.csv'))
        print(" - labeled (optional) :", base.replace('.csv', '_with_labels.csv'))
        print(" - summary+labels     :", base.replace('.csv', '_topic_summary_with_labels.csv'))
        return base

    run_index = []

    # (a) 통합(POOLED)
    if args.topic_scope in ("pooled", "both"):
        pooled_name = args.filename_template.format(scope="pooled", addr_slug="all-stores")
        pooled_path = out_dir / pooled_name
        base = run_topic("pooled", None, pooled_path)
        run_index.append({
            "scope": "pooled",
            "business_id": "ALL",
            "business_address": "ALL",
            "n_sentences": int(len(sent_df)),
            "n_reviews": int(sent_df["review_id"].nunique()) if "review_id" in sent_df.columns else None,
            "status": "modeled",
            "output_base": base
        })

    # (b) 점포별(임계치 필터 적용)
    if args.topic_scope in ("per-store", "both"):
        MIN_S = int(args.per_store_min_sentences)
        MIN_R = int(args.per_store_min_reviews)

        for bid in biz_ids:
            n_s = int(sent_counts.get(bid, 0))
            n_r = int(review_counts.get(bid, 0)) if (review_counts is not None) else None

            if n_s < MIN_S:
                print(f"[Skip] {bid} → sentences={n_s} < {MIN_S}")
                run_index.append({
                    "scope":"per-store","business_id":bid,
                    "business_address": (id2meta.get(bid, {}) or {}).get("address",""),
                    "n_sentences":n_s,"n_reviews":n_r,"status":"skipped_low_sentences","output_base":None
                })
                continue
            if (review_counts is not None) and (n_r < MIN_R):
                print(f"[Skip] {bid} → reviews={n_r} < {MIN_R}")
                run_index.append({
                    "scope":"per-store","business_id":bid,
                    "business_address": (id2meta.get(bid, {}) or {}).get("address",""),
                    "n_sentences":n_s,"n_reviews":n_r,"status":"skipped_low_reviews","output_base":None
                })
                continue

            addr_slug = addr_slug_of(bid)
            fname = args.filename_template.format(scope="perstore", addr_slug=addr_slug)
            out_path = out_dir / fname
            base = run_topic("per-store", bid, out_path)
            run_index.append({
                "scope":"per-store","business_id":bid,
                "business_address": (id2meta.get(bid, {}) or {}).get("address",""),
                "n_sentences":n_s,"n_reviews":n_r,"status":"modeled","output_base":base
            })

    # 5) 실행 인덱스 저장(옵션)
    if args.write_index and run_index:
        idx_path = out_dir / "topic_run_index.csv"
        pd.DataFrame(run_index).to_csv(idx_path, index=False, encoding="utf-8")
        print(f"[Index] Wrote summary → {idx_path}")


if __name__ == "__main__":
    main()
