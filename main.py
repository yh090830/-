import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------
# 기본 설정
# ---------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온 데이터를 이용해 연도에 따른 기온 변화를 분석하고 미래 기온을 예측합니다.")

# ---------------------------------------
# 데이터 불러오기
# ---------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜를 날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자형 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception as e:
    st.error("서울 기온 데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# ---------------------------------------
# 2025년 이후 데이터 제외
# ---------------------------------------
df = df[df["연도"] <= 2025].copy()

# ---------------------------------------
# 연도별 관측일 수 계산
# 평균기온이 실제로 존재하는 날짜만 관측일로 계산
# ---------------------------------------
year_count = (
    df.dropna(subset=["평균기온"])
      .groupby("연도")
      .size()
      .reset_index(name="관측일수")
)

# ---------------------------------------
# 연도별 평균기온 계산
# 관측일이 300일 이상인 해만 사용
# ---------------------------------------
year_temp = (
    df.dropna(subset=["평균기온"])
      .groupby("연도", as_index=False)["평균기온"]
      .mean()
      .rename(columns={"평균기온": "연평균기온"})
)

year_temp = year_temp.merge(year_count, on="연도", how="left")

# 300일 미만인 해 제거
year_temp = year_temp[year_temp["관측일수"] >= 300].copy()

# 연도순 정렬
year_temp = year_temp.sort_values("연도").reset_index(drop=True)

# ---------------------------------------
# 회귀분석
# 독립변수 = 1908년부터 지난 연수
# x = 연도 - 1908
# ---------------------------------------
year_temp["지난연수"] = year_temp["연도"] - 1908

x = year_temp["지난연수"].to_numpy()
y = year_temp["연평균기온"].to_numpy()

# 1차 회귀 직선
slope, intercept = np.polyfit(x, y, 1)

# 회귀 예측값
year_temp["회귀기온"] = slope * x + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# ---------------------------------------
# 회귀선을 그릴 연도 범위
# 1908 ~ 2100
# ---------------------------------------
prediction_years = np.arange(1908, 2101)
prediction_x = prediction_years - 1908
prediction_temps = slope * prediction_x + intercept

# ---------------------------------------
# 기본 정보 표시
# ---------------------------------------
start_year = int(year_temp["연도"].min())
end_year = int(year_temp["연도"].max())
data_count = len(year_temp)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{data_count}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")

st.caption(
    f"2025년까지의 데이터 중 관측일이 300일 이상인 연도만 사용했습니다. "
    f"회귀식의 독립변수는 '연도 - 1908'입니다."
)

# ---------------------------------------
# 연도 선택 슬라이더
# ---------------------------------------
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 선택한 연도의 예상 기온
selected_x = selected_year - 1908
predicted_temp = slope * selected_x + intercept

st.markdown("### 🔮 선택한 연도의 예상 평균기온")

st.metric(
    label=f"{selected_year}년 예상 평균기온",
    value=f"{predicted_temp:.2f} °C"
)

# ---------------------------------------
# Plotly 산점도 + 회귀선
# ---------------------------------------
fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=year_temp["연도"],
        y=year_temp["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=[
            f"{year}년<br>"
            f"연평균기온: {temp:.2f}°C<br>"
            f"관측일수: {days}일"
            for year, temp, days
            in zip(
                year_temp["연도"],
                year_temp["연평균기온"],
                year_temp["관측일수"]
            )
        ],
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=7)
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3)
    )
)

# 선택한 연도의 예측 위치
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=14, symbol="diamond"),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 평균기온: {predicted_temp:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연평균기온과 연도에 따른 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        tickmode="linear",
        dtick=10,
        range=[1900, 2100]
    ),
    hovermode="closest",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# 회귀식과 상관계수
# ---------------------------------------
st.markdown("### 📊 분석 결과")

st.write(
    f"**회귀식:** 연평균기온 = "
    f"{slope:.4f} × (연도 - 1908) + {intercept:.4f}"
)

st.write(
    f"**상관계수:** {correlation:.4f}"
)

if correlation > 0:
    st.write(
        "연도가 증가할수록 연평균기온이 높아지는 경향이 나타납니다."
    )
elif correlation < 0:
    st.write(
        "연도가 증가할수록 연평균기온이 낮아지는 경향이 나타납니다."
    )
else:
    st.write(
        "연도와 연평균기온 사이에 뚜렷한 선형 관계가 나타나지 않습니다."
    )

# ---------------------------------------
# 사용한 데이터 안내
# ---------------------------------------
with st.expander("사용한 데이터 확인"):
    st.write(
        f"- 2025년 이후 데이터: 제외\n"
        f"- 관측일이 300일 미만인 연도: 제외\n"
        f"- 회귀에 사용한 연도: {start_year}년 ~ {end_year}년\n"
        f"- 회귀에 사용한 연도 수: {data_count}개\n"
        f"- 회귀 독립변수: 연도 - 1908"
    )

    st.dataframe(
        year_temp[
            ["연도", "관측일수", "연평균기온", "지난연수", "회귀기온"]
        ],
        use_container_width=True
    )
