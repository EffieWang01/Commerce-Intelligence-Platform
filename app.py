import os

import pandas as pd
import requests
import streamlit as st


API = os.getenv("COMMERCE_API_URL", "http://127.0.0.1:8008")
st.set_page_config(page_title="Commerce Intelligence Platform", page_icon="🛍️", layout="wide")
st.title("Commerce Intelligence Platform")
st.caption("搜索 · 推荐 · 广告 · LLM · Agent 的统一可观测 Demo")

mode = st.sidebar.radio("入口", ["智能导购 Agent", "商品搜索", "个性化推荐"])
user_id = st.sidebar.selectbox("用户", ["u001", "u002", "u003", "新用户"])
user_id = None if user_id == "新用户" else user_id

try:
    health = requests.get(f"{API}/health", timeout=3).json()
    st.sidebar.success(f"{health['profile']} · {health['llm']}")
except requests.RequestException:
    st.error(f"API 未启动：{API}")
    st.code("uvicorn commerce_ai.api:app --port 8008")
    st.stop()

if mode == "智能导购 Agent":
    value = st.text_input("需求", "帮我找预算 500 的降噪耳机，并说明推荐理由")
    endpoint, payload = "/v1/agent", {"message": value, "user_id": user_id}
elif mode == "商品搜索":
    value = st.text_input("搜索词", "降噪耳机")
    endpoint, payload = "/v1/search", {"query": value, "user_id": user_id, "top_k": 10, "include_ads": True}
else:
    value = user_id or "u001"
    st.write(f"为用户 **{value}** 生成推荐")
    endpoint, payload = "/v1/recommend", {"user_id": value, "top_k": 10, "include_ads": True}

if st.button("运行完整链路", type="primary"):
    response = requests.post(API + endpoint, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    if result.get("answer"):
        st.success(result["answer"])
    items = result.get("items", [])
    if items:
        columns = [key for key in ["item_id", "title", "category", "price", "score", "retrieval_score",
                                    "recommendation_score", "ad_score", "sponsored", "reason"] if key in items[0]]
        st.dataframe(pd.DataFrame(items)[columns], hide_index=True, use_container_width=True)
    st.subheader("可观测执行轨迹")
    st.json(result.get("trace", []))

