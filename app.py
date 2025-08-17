# app.py — End-to-end pipeline + dashboard + LLM report (address-based filenames, no biz_id in filenames)

import os
import io
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI

# --- optional: .env 지원 ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== Project modules (your existing code) =====
from config import (
    PATH_BUSINESS, PATH_REVIEW,
    PATH_REVIEWS_FILTERED, PATH_SENT_WITH_SENT, PATH_WITH_TOPICS,
    SENTIMENT_MODEL, EMBEDDING_MODEL
)
from modules.find_business_ids import find_business_ids
from modules.filter_reviews import filter_reviews_by_business_ids
from modules.sentence_sentiment import run_sentence_sentiment
from modules.topic_model_final import apply_bertopic_for_business
from modules.business_meta import load_business_meta  # ← 주소/이름 메타

# ========= App config =========
st.set_page_config(page_title="Business Topic+Sentiment Reporter", page_icon="📊", layout="wide")
API_KEY = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
MODEL_ID = os.getenv("OPENAI_MODEL", "gpt-4o-mini") or st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

# ========= helpers =========
def slugify(s: str, max_len: int = 60) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "store"

def unique_addr_slugger(id2meta: Dict[str, dict]):
    """주소 슬러그를 생성하되, 중복이면 -2, -3 붙여 충돌 방지."""
    seen = {}
    cache = {}
    for bid, meta in id2meta.items():
        base = meta.get("address") or meta.get("name") or "store"
        s = slugify(base, max_len=60)
        cnt = seen.get(s, 0) + 1
        seen[s] = cnt
        cache[bid] = s if cnt == 1 else f"{s}-{cnt}"
    return cache  # {bid: addr_slug}

def attach_address_cols(df: pd.DataFrame, id2meta: Dict[str, dict]) -> pd.DataFrame:
    if "business_id" not in df.columns:
        return df
    out = df.copy()
    out["business_address"] = out["business_id"].map(lambda x: (id2meta.get(x, {}) or {}).get("address", ""))
    out["business_name"] = out.get("business_name", pd.Series("", index=out.index))
    # name이 비어있으면 meta name 보강
    if "business_name" in out.columns and out["business_name"].eq("").any():
        out.loc[out["business_name"].eq(""), "business_name"] = out["business_id"].map(
            lambda x: (id2meta.get(x, {}) or {}).get("name", "")
        )
    return out

def restrict_recent_months(df: pd.DataFrame, months_back: int) -> pd.DataFrame:
    if "month" not in df.columns or df.empty:
        return df
    def ym_to_int(m):
        try:
            y, mm = str(m).split("-"); return int(y)*12 + int(mm)
        except Exception:
            return -10**9
    ms = sorted(df["month"].dropna().unique(), key=ym_to_int)
    if not ms: return df
    last_idx = ym_to_int(ms[-1])
    keep = [m for m in ms if (last_idx - ym_to_int(m)) < months_back]
    return df[df["month"].isin(keep)].copy()

def aggregate_sentences(df: pd.DataFrame, gran: str = "M"):
    if df.empty: return df
    if "date" in df.columns and gran != "M":
        t = pd.to_datetime(df["date"], errors="coerce")
        if gran == "W": df["month"] = t.dt.to_period("W").astype(str)
        elif gran == "D": df["month"] = t.dt.date.astype(str)
        else: df["month"] = t.dt.to_period("M").astype(str)
    grp_cols = ["month","topic_id"]
    extras = []
    if "business_id" in df.columns: extras.append("business_id")
    if "business_address" in df.columns: extras.append("business_address")
    grp_cols = extras + grp_cols
    out = df.groupby(grp_cols, dropna=False).agg(
        n_reviews=("sentiment", "size") if "sentiment" in df.columns else ("topic_id","size"),
        n_pos=("sentiment", lambda x: int(np.sum(x==1)) if "sentiment" in df.columns else 0),
        n_neg=("sentiment", lambda x: int(np.sum(x==0)) if "sentiment" in df.columns else 0),
        n_neu=("sentiment", lambda x: int(np.sum(x==-1)) if "sentiment" in df.columns else 0),
        avg_prob_pos=("sentiment_conf", "mean") if "sentiment_conf" in df.columns else ("topic_id","size"),
        avg_stars=("stars", "mean") if "stars" in df.columns else ("topic_id","size"),
    ).reset_index()
    denom = (out["n_pos"] + out["n_neg"]).replace(0, np.nan)
    out["pos_ratio"] = (out["n_pos"]/denom).clip(0,1)
    out["neg_ratio"] = (out["n_neg"]/denom).clip(0,1)
    out["neu_ratio"] = (out["n_neu"]/(out["n_pos"]+out["n_neg"]+out["n_neu"]).replace(0,np.nan)).clip(0,1)
    return out

def attach_labels(base: pd.DataFrame, sent_df: pd.DataFrame, sum_df: pd.DataFrame, label_priority: str) -> pd.DataFrame:
    base = base.copy()
    lab_sent = sent_df.groupby("topic_id")["topic_label"].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "")
    base = base.merge(lab_sent.rename("topic_label_s"), on="topic_id", how="left")
    if "topic_label" in sum_df.columns:
        lab_sum = sum_df.groupby("topic_id")["topic_label"].agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else "")
        base = base.merge(lab_sum.rename("topic_label_u"), on="topic_id", how="left")
    if "keywords" in sum_df.columns:
        kw_map = sum_df.groupby("topic_id")["keywords"].agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else "")
        base = base.merge(kw_map.rename("keywords"), on="topic_id", how="left")

    def pick(row):
        if label_priority.startswith("topic_label"):
            return row.get("topic_label_s") or row.get("topic_label_u") or row.get("keywords") or str(row.get("topic_id"))
        if label_priority.startswith("label"):
            return row.get("topic_label_u") or row.get("topic_label_s") or row.get("keywords") or str(row.get("topic_id"))
        if label_priority.startswith("keywords"):
            return row.get("keywords") or row.get("topic_label_s") or row.get("topic_label_u") or str(row.get("topic_id"))
        return str(row.get("topic_id"))
    base["topic_label"] = base.apply(pick, axis=1)
    return base

def build_llm_table(df_aggr: pd.DataFrame, k_topics: int, min_cells: int, max_topics_for_prompt: int,
                    store_filter: Optional[List[str]] = None, id2addr: Optional[Dict[str,str]] = None) -> pd.DataFrame:
    sub = df_aggr.copy()
    if store_filter and "business_id" in sub.columns:
        sub = sub[sub["business_id"].isin(store_filter)]
    sub = sub[sub["n_reviews"] >= min_cells]
    vol = sub.groupby("topic_id")["n_reviews"].sum().sort_values(ascending=False)
    top_ids = vol.head(k_topics).index.tolist()
    sub = sub[sub["topic_id"].isin(top_ids)].copy()
    def tscore(g):
        months = sorted(g["month"].unique())
        if len(months) < 6: return np.nan
        def ym_to_int(m): y, mm = str(m).split("-"); return int(y)*12+int(mm)
        months_sorted = sorted(months, key=ym_to_int)
        last3 = months_sorted[-3:]; prev3 = months_sorted[-6:-3]
        m1 = g[g["month"].isin(last3)]["pos_ratio"].mean()
        m0 = g[g["month"].isin(prev3)]["pos_ratio"].mean()
        return (m1 - m0) if (pd.notna(m1) and pd.notna(m0)) else np.nan
    tlist = [(tid, tscore(g)) for tid, g in sub.groupby("topic_id")]
    sub = sub.merge(pd.DataFrame(tlist, columns=["topic_id","trend_score"]), on="topic_id", how="left")
    chosen = sub.groupby("topic_id")["n_reviews"].sum().reset_index().sort_values("n_reviews", ascending=False)["topic_id"].head(max_topics_for_prompt).tolist()
    sub = sub[sub["topic_id"].isin(chosen)].copy()
    # 주소 보강(프롬프트 가독성)
    if id2addr and "business_id" in sub.columns and "business_address" not in sub.columns:
        sub["business_address"] = sub["business_id"].map(id2addr)
    return sub

def table_to_text(tbl: pd.DataFrame) -> str:
    cols = ["business_id","business_address","month","topic_id","topic_label","n_reviews","pos_ratio","neg_ratio","neu_ratio","avg_stars","avg_prob_pos","trend_score"]
    exist = [c for c in cols if c in tbl.columns]
    tmp = tbl[exist].copy()
    for c in ["pos_ratio","neg_ratio","neu_ratio","avg_stars","avg_prob_pos","trend_score"]:
        if c in tmp.columns: tmp[c] = tmp[c].astype(float).round(3)
    lines = [",".join(exist)]
    for _, r in tmp.iterrows():
        lines.append(",".join("" if pd.isna(r[c]) else str(r[c]) for c in exist))
    return "\n".join(lines)

def llm_messages(table_text: str, pooled: bool, store_names: List[str]) -> List[Dict]:
    scope_text = "pooled across stores" if pooled else f"stores: {', '.join(store_names)}"
    sys = "You are a meticulous data analyst. Produce an executive-ready report in structured Markdown."
    user = f"""
Scope: {scope_text}
Columns: business_id, business_address, month, topic_id, topic_label, n_reviews, pos_ratio, neg_ratio, neu_ratio, avg_stars, avg_prob_pos, trend_score

Tasks:
1) Executive summary in 5–8 bullets.
2) Top 5 positive-rising topics (cite months & plausible reasons).
3) Top 5 negative-rising topics (actionable remedies).
4) Dimensions (service/speed/cleanliness/taste/price if visible): drift last 3m vs prev 3m.
5) Flag anomalies (spikes) with months.
6) Action plan ≤6 items with measurable KPI.

Constraints:
- Use only the table; do not invent numbers. Use YYYY-MM.
- ≤600 words; sections: Summary / Positive Momentum / Negative Momentum / Dimensions / Anomalies / Action Plan.

Data:
{table_text}
""".strip()
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def call_llm(messages: List[Dict], temperature: float) -> str:
    if not API_KEY:
        st.warning("OPENAI_API_KEY not set; skipping LLM call.")
        return ""
    client = OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=temperature)
    return resp.choices[0].message.content

# ========= Sidebar – pipeline controls =========
st.sidebar.header("Business selection")
col_a, col_b = st.sidebar.columns(2)
name_substr = col_a.text_input("name-substr", value="")
state = col_b.text_input("state", value="")
city = st.sidebar.text_input("city", value="")
category = st.sidebar.text_input("category", value="Restaurants")
limit = st.sidebar.number_input("limit", min_value=1, max_value=200, value=20, step=1)
biz_ids_text = st.sidebar.text_area("biz-id (one per line; optional)", value="")

st.sidebar.header("Topic scope & thresholds")
topic_scope = st.sidebar.selectbox("topic-scope", ["per-store","pooled","both"], index=0)
per_store_min_sentences = st.sidebar.number_input("per-store min sentences", min_value=0, value=60, step=10)
per_store_min_reviews = st.sidebar.number_input("per-store min reviews (0=ignore)", min_value=0, value=0, step=5)

st.sidebar.header("Topic params & labels")
min_topic_size = st.sidebar.number_input("min-topic-size (0=auto)", min_value=0, value=0, step=5)
nr_topics = st.sidebar.number_input("max-topics (0=None)", min_value=0, value=0, step=2)
enable_zeroshot = st.sidebar.checkbox("enable zeroshot labels", value=True)
label_mode = st.sidebar.selectbox("label-mode", ["core","auto","full"], index=1)
zeroshot_min_prob = st.sidebar.slider("zeroshot min_prob", 0.2, 0.8, 0.4, 0.05)
zeroshot_online = st.sidebar.checkbox("zeroshot allow online model download", value=False)

st.sidebar.header("Output naming")
out_dir_inp = st.sidebar.text_input("out-dir", value=str(Path(PATH_WITH_TOPICS).parent))
# 파일명 템플릿: biz_id 제거, 주소 기반만 사용
filename_template = st.sidebar.text_input("filename-template", value="with_topics__{scope}__{addr_slug}.csv")

st.sidebar.header("Time & LLM")
months_back = st.sidebar.slider("months_back", 3, 36, 12)
time_gran = st.sidebar.selectbox("time granularity", ["M","W","D"], index=0)
label_priority = st.sidebar.selectbox("label priority", ["topic_label (zeroshot)","label (summary)","keywords (fallback)","topic_id only"], index=0)
k_topics = st.sidebar.slider("Top-K topics for charts", 3, 40, 12)
min_cells = st.sidebar.slider("Min rows per topic-month (LLM/pruning)", 1, 200, 10)
run_llm = st.sidebar.checkbox("Generate LLM report", value=False)
max_topics_for_prompt = st.sidebar.slider("Max topics to LLM", 5, 30, 15)
temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)

# ========= Main – run pipeline =========
st.title("Business Topic+Sentiment Reporter")

run_pipeline = st.button("Run pipeline (find → filter → sentiment → topic)")

if run_pipeline:
    with st.status("Running pipeline...", expanded=True) as status:
        # 1) business ids
        if biz_ids_text.strip():
            biz_ids = [line.strip() for line in biz_ids_text.strip().splitlines() if line.strip()]
            st.write(f"Using user-provided biz_ids: {len(biz_ids)}")
        else:
            biz_ids = find_business_ids(
                business_json_path=str(PATH_BUSINESS),
                name_substring=name_substr or None,
                category_keyword=category or "Restaurants",
                city=city or None,
                state=state or None,
                limit=int(limit)
            )
            st.write(f"[Picked business_id(s)] {biz_ids}")
        if not biz_ids:
            st.error("No business matched.")
            st.stop()

        # 1.5) 주소/이름 메타 로드 + addr_slug(충돌 방지 포함)
        id2meta = load_business_meta(str(PATH_BUSINESS), biz_ids)  # {bid:{name,address}}
        id2addr = {bid: (m or {}).get("address","") for bid, m in id2meta.items()}
        id2slug = unique_addr_slugger(id2meta)  # {bid: addr_slug}

        # 2) filter reviews → JSONL
        kept = filter_reviews_by_business_ids(
            review_json_path=str(PATH_REVIEW),
            target_business_ids=biz_ids,
            out_jsonl_path=str(PATH_REVIEWS_FILTERED),
        )
        st.write(f"[Filtered reviews] kept={kept} → {PATH_REVIEWS_FILTERED}")

        # 3) sentence sentiment
        run_sentence_sentiment(
            filtered_jsonl_path=str(PATH_REVIEWS_FILTERED),
            out_csv_path=str(PATH_SENT_WITH_SENT),
            sentiment_model=SENTIMENT_MODEL,
            max_chars=256,
            batch_size=64
        )
        st.write(f"[Sentence sentiment] → {PATH_SENT_WITH_SENT}")

        # 3.5) per-store counts (thresholds)
        sent_df = pd.read_csv(PATH_SENT_WITH_SENT)
        sent_counts = sent_df.groupby("business_id").size().to_dict()
        if per_store_min_reviews > 0 and "review_id" in sent_df.columns:
            review_counts = (
                sent_df.drop_duplicates(["business_id","review_id"])
                       .groupby("business_id").size().to_dict()
            )
        else:
            review_counts = {}

        # 4) output dir
        out_dir = Path(out_dir_inp).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        st.write(f"[Output dir] {out_dir}")

        def run_topic(scope: str, business_id, out_base_path: Path):
            _ = apply_bertopic_for_business(
                sentences_csv_path=str(PATH_SENT_WITH_SENT),
                out_csv_path=str(out_base_path),
                business_id=business_id,                # None/"__ALL__" → pooled
                embedding_model=EMBEDDING_MODEL,
                min_topic_size=(None if int(min_topic_size)==0 else int(min_topic_size)),
                nr_topics=(None if int(nr_topics)==0 else int(nr_topics)),
                enable_zeroshot=bool(enable_zeroshot),
                zeroshot_online=bool(zeroshot_online),
                zeroshot_min_prob=float(zeroshot_min_prob),
                label_mode=label_mode,
            )
            base = str(out_base_path)
            st.write(f"[BERTopic done] scope={scope} business_id={business_id} → {base}")
            st.write("Artifacts:")
            st.write(" - per-sentence topics:", base)
            st.write(" - topic summary      :", base.replace('.csv', '_topic_summary.csv'))
            st.write(" - topic trend monthly:", base.replace('.csv', '_topic_trend_by_month.csv'))
            st.write(" - topic examples     :", base.replace('.csv', '_topic_examples.csv'))
            st.write(" - labeled (optional) :", base.replace('.csv', '_with_labels.csv'))
            st.write(" - summary+labels     :", base.replace('.csv', '_topic_summary_with_labels.csv'))
            return base

        # pooled
        if topic_scope in ("pooled","both"):
            pooled_name = filename_template.format(scope="pooled", addr_slug="all-stores")
            run_topic("pooled", None, out_dir / pooled_name)

        # per-store with thresholds
        if topic_scope in ("per-store","both"):
            for bid in biz_ids:
                n_s = int(sent_counts.get(bid, 0))
                n_r = int(review_counts.get(bid, 0)) if review_counts else None
                if n_s < int(per_store_min_sentences):
                    st.write(f"[Skip] {bid} → sentences={n_s} < {per_store_min_sentences}")
                    continue
                if (per_store_min_reviews > 0) and (n_r is not None) and (n_r < int(per_store_min_reviews)):
                    st.write(f"[Skip] {bid} → reviews={n_r} < {per_store_min_reviews}")
                    continue
                addr_slug = id2slug.get(bid, slugify(bid))
                fname = filename_template.format(scope="perstore", addr_slug=addr_slug)
                run_topic("per-store", bid, out_dir / fname)

        status.update(label="Pipeline finished.", state="complete", expanded=False)

# ========= Post-run: load outputs & visualize =========
st.subheader("Load and visualize outputs")
scan_dir = st.text_input("Scan directory for outputs", value=str(Path(out_dir_inp)))
scan = st.button("Scan & Load")

def read_csv_safe(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Failed to read {p.name}: {e}")
    return None

def artifact_paths(base_csv: Path) -> Dict[str, Path]:
    b = str(base_csv)
    return {
        "base": Path(b),
        "summary": Path(b.replace(".csv", "_topic_summary.csv")),
        "trend": Path(b.replace(".csv", "_topic_trend_by_month.csv")),
        "examples": Path(b.replace(".csv", "_topic_examples.csv")),
        "with_labels": Path(b.replace(".csv", "_with_labels.csv")),
        "summary_with_labels": Path(b.replace(".csv", "_topic_summary_with_labels.csv")),
    }

if scan:
    load_dir = Path(scan_dir)
    if not load_dir.exists():
        st.error("Directory not found.")
        st.stop()

    bases = sorted(load_dir.glob("with_topics__*.csv"))
    if not bases:
        st.warning("No base CSVs found (pattern: with_topics__*.csv).")
        st.stop()

    sent_list, sum_list, tr_list = [], [], []
    for b in bases:
        ap = artifact_paths(b)
        df_base = read_csv_safe(ap["base"])
        if df_base is not None:
            sent_list.append(df_base)
        df_sum = read_csv_safe(ap["summary_with_labels"]) or read_csv_safe(ap["summary"])
        if df_sum is not None:
            sum_list.append(df_sum)
        df_tr = read_csv_safe(ap["trend"])
        if df_tr is not None:
            tr_list.append(df_tr)

    sent_df = pd.concat(sent_list, ignore_index=True) if sent_list else pd.DataFrame()
    sum_df = pd.concat(sum_list, ignore_index=True) if sum_list else pd.DataFrame()
    trend_df = pd.concat(tr_list, ignore_index=True) if tr_list else pd.DataFrame()

    if sent_df.empty:
        st.warning("No per-sentence outputs loaded; only limited views available.")
        st.stop()

    # 주소 주입(산출물에 business_address가 없는 경우 보강)
    if "business_address" not in sent_df.columns and "business_id" in sent_df.columns:
        uniq_ids = sent_df["business_id"].dropna().unique().tolist()
        id2meta = load_business_meta(str(PATH_BUSINESS), uniq_ids)
        id2addr = {bid: (m or {}).get("address","") for bid, m in id2meta.items()}
        sent_df = attach_address_cols(sent_df, id2meta)
    else:
        id2addr = None

    st.success(f"Loaded sentences: {len(sent_df):,} rows | stores: {sent_df['business_id'].nunique() if 'business_id' in sent_df.columns else 0}")

    # month 보정
    if "date" in sent_df.columns and "month" not in sent_df.columns:
        sent_df["date"] = pd.to_datetime(sent_df["date"], errors="coerce")
        sent_df["month"] = sent_df["date"].dt.to_period("M").astype(str)

    # 노이즈 토픽 제외 옵션
    include_noise = st.checkbox("Include noise topic (-1)", value=False)
    if not include_noise and "topic_id" in sent_df.columns:
        sent_df = sent_df[sent_df["topic_id"] != -1]

    # 집계 & 최근 윈도우
    aggr = aggregate_sentences(sent_df, gran=time_gran)
    aggr_recent = restrict_recent_months(aggr, months_back)
    aggr_recent = attach_labels(aggr_recent, sent_df, sum_df, label_priority)

    # 스토어 목록(주소 표시)
    stores = []
    if "business_id" in aggr_recent.columns:
        if "business_address" in aggr_recent.columns:
            stores = sorted(aggr_recent[["business_id","business_address"]].drop_duplicates().itertuples(index=False, name=None), key=lambda x: x[1])
        else:
            stores = [(bid, bid) for bid in sorted(aggr_recent["business_id"].dropna().unique())]

    tab_over, tab_topics, tab_stores, tab_trends, tab_llm = st.tabs(["Overview", "Topics", "Stores", "Trends", "LLM Report"])

    with tab_over:
        st.markdown("#### Volume by Topic (recent window)")
        vol = (aggr_recent
               .groupby(["topic_id","topic_label"])["n_reviews"]
               .sum().reset_index().sort_values("n_reviews", ascending=False))
        st.dataframe(vol.head(50), use_container_width=True)

        st.markdown("#### Top-K pos_ratio time series")
        top_ids = vol["topic_id"].head(int(k_topics)).tolist()
        plot_df = aggr_recent[aggr_recent["topic_id"].isin(top_ids)].copy()
        fig = px.line(plot_df.sort_values("month"), x="month", y="pos_ratio", color="topic_label", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab_topics:
        st.markdown("#### Topic detail (recent)")
        cols = ["business_address","business_id","month","topic_id","topic_label","n_reviews","pos_ratio","neg_ratio","neu_ratio","avg_stars","avg_prob_pos"]
        cols = [c for c in cols if c in aggr_recent.columns]
        st.dataframe(aggr_recent.sort_values(["n_reviews"], ascending=False)[cols].head(1000), use_container_width=True)

    with tab_stores:
        if "business_id" in aggr_recent.columns:
            st.markdown("#### Store × Topic matrix (volume)")
            cols = ["business_address","business_id","topic_id","topic_label","n_reviews"]
            cols = [c for c in cols if c in aggr_recent.columns]
            m = (aggr_recent.groupby([c for c in cols if c in ["business_address","business_id","topic_id","topic_label"]])["n_reviews"]
                 .sum().reset_index().sort_values(["business_address" if "business_address" in cols else "business_id","n_reviews"], ascending=[True, False]))
            st.dataframe(m.head(2000), use_container_width=True)

            st.markdown("#### Worst stores per negative topic (recent)")
            neg = aggr_recent.copy()
            neg = neg[neg["n_reviews"] >= int(min_cells)]
            neg["neg_rate"] = neg["neg_ratio"].astype(float)
            worst = (neg.sort_values(["topic_id","neg_rate","n_reviews"], ascending=[True, False, False])
                        .groupby("topic_id").head(5))
            cols2 = ["business_address","business_id","topic_id","topic_label","n_reviews","neg_rate","pos_ratio","avg_stars"]
            cols2 = [c for c in cols2 if c in worst.columns]
            st.dataframe(worst[cols2], use_container_width=True)
        else:
            st.info("business_id 컬럼이 없어 점포 비교 표는 생략합니다.")

    with tab_trends:
        st.markdown("#### Monthly counts by topic (recent)")
        cnt = (aggr_recent.groupby(["month","topic_id","topic_label"])["n_reviews"]
               .sum().reset_index().sort_values(["month","n_reviews"], ascending=[True,False]))
        fig2 = px.bar(cnt[cnt["topic_id"].isin(vol["topic_id"].head(int(k_topics)))],
                      x="month", y="n_reviews", color="topic_label", barmode="stack")
        st.plotly_chart(fig2, use_container_width=True)

    with tab_llm:
        st.markdown("#### Table passed to LLM")
        pooled_mode = st.checkbox("Report as pooled across selected rows", value=True)
        store_filter = None
        store_names = []
        if not pooled_mode and stores:
            # 주소로 보여 주고, 내부적으로 business_id를 필터링
            display_options = [f"{addr} [{bid}]" for (bid, addr) in stores]
            sel = st.multiselect("Choose stores for the report", options=display_options, default=display_options[:1])
            sel_ids = []
            sel_names = []
            for item in sel:
                # 끝의 [bid] 추출
                m = re.search(r"\[([^\]]+)\]\s*$", item)
                if m:
                    sel_ids.append(m.group(1))
                name_part = item.rsplit(" [", 1)[0]
                sel_names.append(name_part)
            store_filter = sel_ids
            store_names = sel_names

        # id→address 사전 준비
        if "business_id" in aggr_recent.columns:
            uniq_ids = aggr_recent["business_id"].dropna().unique().tolist()
            id2meta = load_business_meta(str(PATH_BUSINESS), uniq_ids)
            id2addr = {bid: (m or {}).get("address","") for bid, m in id2meta.items()}
        else:
            id2addr = None

        tbl = build_llm_table(
            aggr_recent, k_topics=int(k_topics), min_cells=int(min_cells),
            max_topics_for_prompt=int(max_topics_for_prompt),
            store_filter=store_filter, id2addr=id2addr
        )
        st.dataframe(tbl.head(500), use_container_width=True)

        if run_llm:
            table_text = table_to_text(tbl)
            msgs = llm_messages(table_text, pooled=pooled_mode, store_names=(store_names if not pooled_mode else []))
            with st.spinner("Generating LLM report..."):
                report_md = call_llm(msgs, temperature=float(temperature))
            if report_md:
                st.subheader("LLM Report")
                st.markdown(report_md)
                b = io.BytesIO(report_md.encode("utf-8"))
                st.download_button("Download report.md", b, file_name="report.md", mime="text/markdown")
            else:
                st.info("No report generated (missing API key or empty table).")
