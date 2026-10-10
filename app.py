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
st.caption("搜索 · 推荐 · 广告 · LLM · Agent 的统一可观测 Demo")

mode = st.sidebar.radio("入口", ["智能导购 Agent", "商品搜索", "个性化推荐"])
user_id = st.sidebar.selectbox("用户", ["u001", "u002", "u003", "新用户"])
user_id = None if user_id == "新用户" else user_id
personalize = st.sidebar.checkbox("启用个性化", value=False, disabled=mode == "个性化推荐")
if mode != "个性化推荐" and not personalize:
    user_id = None

try:
    health = requests.get(f"{API}/health", timeout=3).json()
    st.sidebar.success(f"{health['profile']} · {health['providers']['llm']}")
    catalog = health.get("catalog", {})
    if catalog:
        st.sidebar.caption(
            f"商品目录：{catalog.get('products', 0)} 件 · "
            f"{len(catalog.get('categories', []))} 个品类"
        )
    if health.get("warnings"):
        st.sidebar.warning("部分服务已降级：" + ", ".join(health["warnings"]))
except requests.RequestException:
    st.error(f"API 未启动：{API}")
    st.code("uvicorn commerce_ai.api:app --port 8008")
    st.stop()

if mode == "智能导购 Agent":
    value = st.text_input("需求", "帮我找预算 500 的降噪耳机，并说明推荐理由")
    endpoint, payload = "/v1/agent", {"message": value, "user_id": user_id}
elif mode == "商品搜索":
    category = st.selectbox("品类", ["全部", *catalog.get("categories", [])])
    value = st.text_input("搜索词", "降噪耳机")
    search_query = value if category == "全部" else f"{category} {value}"
    endpoint, payload = "/v1/search", {"query": value, "user_id": user_id, "top_k": 10, "include_ads": True}
    payload["query"] = search_query
else:
    value = user_id or "u001"
    st.write(f"为用户 **{value}** 生成推荐")
    endpoint, payload = "/v1/recommend", {"user_id": value, "top_k": 10, "include_ads": True}

if mode != "个性化推荐":
    uploaded = st.file_uploader("商品图片（可选）", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        mime = uploaded.type or "image/jpeg"
        payload["image_data"] = f"data:{mime};base64," + base64.b64encode(uploaded.getvalue()).decode()

if st.button("运行完整链路", type="primary"):
    response = requests.post(API + endpoint, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    if result.get("answer"):
        st.success(result["answer"])
    items = result.get("items", [])
    if items:
        columns = [key for key in [
            "image_url", "item_id", "title", "category", "price", "source", "score", "retrieval_score",
            "recommendation_score", "multimodal_score", "rerank_score", "ad_score",
            "sponsored", "reason",
        ] if key in items[0]]
        table = pd.DataFrame(items)[columns]
        if "price" in table:
            currencies = pd.DataFrame(items).get("currency", pd.Series([None] * len(items)))
            table["price"] = [
                format_price(value, currency)
                for value, currency in zip(table["price"], currencies)
            ]
        st.dataframe(
            table,
            column_config={"image_url": st.column_config.ImageColumn("图片", width="small")},
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("当前商品目录中没有足够相关的结果。")
    st.subheader("可观测执行轨迹")
    st.json(result.get("trace", []))

