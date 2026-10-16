import base64
import os

import pandas as pd
import requests
import streamlit as st


API = os.getenv("COMMERCE_API_URL", "http://127.0.0.1:8008")


def format_price(value, currency):
    if pd.isna(value):
        return "未知"
    symbol = {"USD": "$", "CNY": "¥"}.get(currency or "CNY", "")
    return f"{symbol}{value:,.2f}"


st.set_page_config(page_title="Commerce Intelligence Platform", page_icon="🛍️", layout="wide")
st.title("Commerce Intelligence Platform")
st.caption("商品搜索与智能导购")

mode = st.sidebar.radio("入口", ["智能导购 Agent", "商品搜索", "个性化推荐"])
user_id = st.sidebar.selectbox("用户", ["u001", "u002", "u003", "新用户"])
user_id = None if user_id == "新用户" else user_id
personalize = st.sidebar.checkbox("启用个性化", value=False, disabled=mode == "个性化推荐")
if mode != "个性化推荐" and not personalize:
    user_id = None

try:
    health_response = requests.get(f"{API}/health", timeout=10)
    health_response.raise_for_status()
    health = health_response.json()
    st.sidebar.success(f"{health['profile']} · {health['providers']['llm']}")
    catalog = health.get("catalog", {})
    if catalog:
        st.sidebar.caption(
            f"商品目录：{catalog.get('products', 0)} 件 · "
            f"{len(catalog.get('categories', []))} 个品类"
        )
    if health.get("warnings"):
        st.sidebar.warning("部分服务已降级：" + ", ".join(health["warnings"]))
    ranker_name = health["providers"].get("ranking", "unknown")
    st.sidebar.caption("排序：" + {"lightgbm-lambdarank": "LambdaMART", "weighted-rules": "规则基线"}.get(ranker_name, ranker_name))
    st.sidebar.caption("查询理解：" + health["providers"]["llm"])
except (requests.RequestException, ValueError, KeyError):
    st.error("搜索服务暂不可用，请稍后重试。")
    st.stop()

if mode == "智能导购 Agent":
    value = st.text_input("需求", "wireless headphones under 50")
    endpoint, payload = "/v1/agent", {"message": value, "user_id": user_id}
elif mode == "商品搜索":
    category = st.selectbox("品类", ["全部", *catalog.get("categories", [])])
    value = st.text_input("搜索词", "wireless headphones")
    search_query = value if category == "全部" else f"{category} {value}"
    endpoint, payload = "/v1/search", {"query": value, "user_id": user_id, "top_k": 10, "include_ads": True}
    payload["query"] = search_query
else:
    value = user_id or "u001"
    st.write(f"为用户 **{value}** 生成推荐")
    endpoint, payload = "/v1/recommend", {"user_id": value, "top_k": 10, "include_ads": True}

if mode != "个性化推荐":
    image_enabled = (health["providers"].get("multimodal") != "disabled"
                     or health["providers"]["llm"] != "heuristic")
    uploaded = st.file_uploader("商品图片（可选）", type=["jpg", "jpeg", "png", "webp"],
                                disabled=not image_enabled)
    if uploaded:
        mime = uploaded.type or "image/jpeg"
        payload["image_data"] = f"data:{mime};base64," + base64.b64encode(uploaded.getvalue()).decode()

if st.button("生成推荐" if mode == "个性化推荐" else "搜索", type="primary"):
    if mode != "个性化推荐" and not value.strip():
        st.warning("请输入搜索需求。")
        st.stop()
    try:
        with st.spinner("正在查询商品…"):
            response = requests.post(API + endpoint, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
    except requests.Timeout:
        st.error("查询超时，请稍后重试。")
        st.stop()
    except (requests.RequestException, ValueError):
        st.error("查询失败，请稍后重试。")
        st.stop()
    trace = result.get("trace", [])
    parsed = result.get("intent") or next((event.get("output") for event in trace
                                           if event.get("action") == "understand"), None)
    rank_event = next((event for event in trace if event.get("stage") == "rank"), {})
    if rank_event:
        st.caption(f"候选商品 {rank_event['candidates']} 件 · 返回 {rank_event['returned']} 件")
    if parsed:
        with st.expander("查询解析"):
            st.json(parsed)
    if result.get("answer"):
        st.success(result["answer"])
    items = result.get("items", [])
    if items:
        columns = [key for key in [
            "image_url", "title", "price", "score", "category", "reason", "item_id", "source", "retrieval_score",
            "recommendation_score", "multimodal_score", "rerank_score", "ad_score",
            "sponsored",
        ] if key in items[0]]
        table = pd.DataFrame(items)[columns]
        if "price" in table:
            item_frame = pd.DataFrame(items)
            currencies = item_frame.get("currency", item_frame.get("source_currency", pd.Series([None] * len(items))))
            table["price"] = [
                format_price(value, currency)
                for value, currency in zip(table["price"], currencies)
            ]
        st.dataframe(
            table,
            column_config={
                "image_url": st.column_config.ImageColumn("图片", width="small"),
                "title": st.column_config.TextColumn("商品", width="large"),
                "price": st.column_config.TextColumn("价格", width="small"),
                "score": st.column_config.NumberColumn("排序分", format="%.3f", width="small"),
            },
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("当前商品目录中没有足够相关的结果。")
    with st.expander("执行轨迹"):
        st.json(result.get("trace", []))

