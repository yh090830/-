import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 연평균기온을 이용해 선형회귀 모델을 만들고 "
    "학습 기간에 따른 예측 성능을 비교합니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()

except Exception as e:

    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 2025년까지의 데이터만 사용
# =========================================================

df = df[
    (df["연도"] <= 2025)
    & (df["평균기온"].notna())
].copy()


# =========================================================
# 연도별 연평균기온 계산
# =========================================================

yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# =========================================================
# 관측일이 300일 이상인 연도만 사용
# =========================================================

yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# =========================================================
# 독립변수
# 1908년부터 지난 연수
# =========================================================

yearly["지난연수"] = yearly["연도"] - 1908


# =========================================================
# 전체 기간 회귀
# =========================================================

X_all = yearly[["지난연수"]]
y_all = yearly["연평균기온"]

model_all = LinearRegression()
model_all.fit(X_all, y_all)

yearly["전체회귀예측"] = model_all.predict(X_all)

all_slope = model_all.coef_[0]
all_intercept = model_all.intercept_


# =========================================================
# 전체 기간 회귀 성능
# =========================================================

all_mae = mean_absolute_error(
    y_all,
    yearly["전체회귀예측"]
)

all_mse = mean_squared_error(
    y_all,
    yearly["전체회귀예측"]
)

all_r2 = r2_score(
    y_all,
    yearly["전체회귀예측"]
)


# =========================================================
# 학습 / 테스트 데이터 설정
#
# 최근 50년:
# 1956~2005 학습
#
# 최근 100년:
# 1906~2005 학습
#
# 공통 테스트:
# 2006~2025
# =========================================================

train_50 = yearly[
    (yearly["연도"] >= 1956)
    & (yearly["연도"] <= 2005)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906)
    & (yearly["연도"] <= 2005)
].copy()

test = yearly[
    (yearly["연도"] >= 2006)
    & (yearly["연도"] <= 2025)
].copy()


# =========================================================
# 함수: 회귀모델 학습 + 평가
# =========================================================

def make_model(train_data, test_data):

    X_train = train_data[["지난연수"]]
    y_train = train_data["연평균기온"]

    X_test = test_data[["지난연수"]]
    y_test = test_data["연평균기온"]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    train_pred = model.predict(X_train)

    test_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_pred
    )

    mse = mean_squared_error(
        y_test,
        test_pred
    )

    r2 = r2_score(
        y_test,
        test_pred
    )

    return model, train_pred, test_pred, mae, mse, r2


# =========================================================
# 최근 50년 모델
# =========================================================

model_50, train_pred_50, test_pred_50, mae_50, mse_50, r2_50 = (
    make_model(train_50, test)
)


# =========================================================
# 최근 100년 모델
# =========================================================

model_100, train_pred_100, test_pred_100, mae_100, mse_100, r2_100 = (
    make_model(train_100, test)
)


# =========================================================
# 기울기
# =========================================================

slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]


# =========================================================
# 상단 요약
# =========================================================

st.markdown("## 📊 데이터 구성")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "최근 50년 학습",
        f"{train_50['연도'].min()}~{train_50['연도'].max()}",
    )

    st.write(
        f"사용 연도: {len(train_50)}개"
    )


with col2:

    st.metric(
        "최근 100년 학습",
        f"{train_100['연도'].min()}~{train_100['연도'].max()}",
    )

    st.write(
        f"사용 연도: {len(train_100)}개"
    )


with col3:

    st.metric(
        "공통 테스트",
        f"{test['연도'].min()}~{test['연도'].max()}",
    )

    st.write(
        f"사용 연도: {len(test)}개"
    )


st.info(
    "※ 연간 평균기온이 계산되더라도 관측일수가 300일 미만인 연도는 제외했습니다."
)


# =========================================================
# 전체 데이터 회귀
# =========================================================

st.markdown("## 1️⃣ 전체 데이터에 대한 선형회귀")


fig_all = go.Figure()


fig_all.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=[
            f"{year}년<br>"
            f"연평균기온: {temp:.2f}℃<br>"
            f"관측일수: {days}일"
            for year, temp, days
            in zip(
                yearly["연도"],
                yearly["연평균기온"],
                yearly["관측일수"]
            )
        ],
        hovertemplate="%{text}<extra></extra>"
    )
)


fig_all.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["전체회귀예측"],
        mode="lines",
        name="전체 회귀선",
        line=dict(width=3)
    )
)


fig_all.update_layout(
    title="서울 연평균기온과 전체 기간 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    height=550
)


st.plotly_chart(
    fig_all,
    use_container_width=True
)


st.write(
    f"**회귀선 기울기:** {all_slope:.5f} ℃/년"
)

st.write(
    f"**전체 데이터 MAE:** {all_mae:.3f} ℃"
)

st.write(
    f"**전체 데이터 MSE:** {all_mse:.3f} ℃²"
)

st.write(
    f"**전체 데이터 R²:** {all_r2:.3f}"
)


# =========================================================
# 학습 기간 비교
# =========================================================

st.markdown("## 2️⃣ 최근 50년 vs 최근 100년 학습")


comparison = pd.DataFrame(
    {
        "모델": [
            "최근 50년",
            "최근 100년"
        ],
        "학습기간": [
            "1956~2005",
            "1906~2005"
        ],
        "학습연도수": [
            len(train_50),
            len(train_100)
        ],
        "기울기(℃/년)": [
            slope_50,
            slope_100
        ],
        "MAE": [
            mae_50,
            mae_100
        ],
        "MSE": [
            mse_50,
            mse_100
        ],
        "R²": [
            r2_50,
            r2_100
        ]
    }
)


st.dataframe(
    comparison.style.format(
        {
            "기울기(℃/년)": "{:.5f}",
            "MAE": "{:.3f}",
            "MSE": "{:.3f}",
            "R²": "{:.3f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 기울기 비교
# =========================================================

st.markdown("### 📈 회귀선 기울기 비교")

slope_difference = slope_50 - slope_100


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "최근 50년 기울기",
        f"{slope_50:.5f} ℃/년"
    )


with col2:

    st.metric(
        "최근 100년 기울기",
        f"{slope_100:.5f} ℃/년"
    )


with col3:

    st.metric(
        "기울기 차이",
        f"{slope_difference:.5f} ℃/년"
    )


# =========================================================
# 테스트 데이터 실제값 vs 예측값
# =========================================================

st.markdown("## 3️⃣ 공통 테스트 데이터 예측 결과")

test_result = test[
    ["연도", "연평균기온"]
].copy()

test_result["최근50년_예측"] = test_pred_50

test_result["최근100년_예측"] = test_pred_100


fig_test = go.Figure()


fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["연평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)


fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["최근50년_예측"],
        mode="lines",
        name="최근 50년 학습 예측"
    )
)


fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["최근100년_예측"],
        mode="lines",
        name="최근 100년 학습 예측"
    )
)


fig_test.update_layout(
    title="2006~2025 테스트 데이터 실제값과 예측값",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=2
    ),
    height=550
)


st.plotly_chart(
    fig_test,
    use_container_width=True
)


# =========================================================
# 테스트 성능 시각화
# =========================================================

st.markdown("## 4️⃣ 테스트 데이터 예측 성능 비교")


fig_metrics = go.Figure()


fig_metrics.add_trace(
    go.Bar(
        x=["최근 50년", "최근 100년"],
        y=[mae_50, mae_100],
        name="MAE"
    )
)


fig_metrics.update_layout(
    title="MAE 비교 (낮을수록 좋음)",
    xaxis_title="학습 기간",
    yaxis_title="MAE (℃)",
    height=400
)


st.plotly_chart(
    fig_metrics,
    use_container_width=True
)


fig_r2 = go.Figure()


fig_r2.add_trace(
    go.Bar(
        x=["최근 50년", "최근 100년"],
        y=[r2_50, r2_100],
        name="R²"
    )
)


fig_r2.update_layout(
    title="R² 비교 (높을수록 좋음)",
    xaxis_title="학습 기간",
    yaxis_title="R²",
    height=400
)


st.plotly_chart(
    fig_r2,
    use_container_width=True
)


# =========================================================
# 결과 해석
# =========================================================

st.markdown("## 🔎 결과 해석")


if slope_50 > slope_100:

    st.write(
        "최근 50년을 학습한 회귀선의 기울기가 최근 100년을 학습한 "
        "회귀선보다 큽니다. 즉, 최근 기간을 학습했을 때 연도 증가에 "
        "따른 연평균기온의 상승 추세를 더 크게 나타냅니다."
    )

elif slope_50 < slope_100:

    st.write(
        "최근 100년을 학습한 회귀선의 기울기가 최근 50년을 학습한 "
        "회귀선보다 큽니다. 즉, 장기간의 데이터를 사용했을 때 "
        "연도 증가에 따른 상승 추세가 더 크게 나타납니다."
    )

else:

    st.write(
        "두 학습 기간의 회귀선 기울기가 거의 같습니다."
    )


if mae_50 < mae_100:

    st.write(
        f"2006~2025 테스트 데이터에서 최근 50년 모델의 MAE가 "
        f"{mae_50:.3f}℃로 최근 100년 모델의 {mae_100:.3f}℃보다 "
        "작아 평균적인 예측 오차가 더 작았습니다."
    )

elif mae_50 > mae_100:

    st.write(
        f"2006~2025 테스트 데이터에서 최근 100년 모델의 MAE가 "
        f"{mae_100:.3f}℃로 최근 50년 모델의 {mae_50:.3f}℃보다 "
        "작아 평균적인 예측 오차가 더 작았습니다."
    )

else:

    st.write(
        "두 모델의 MAE가 같습니다."
    )


if r2_50 > r2_100:

    st.write(
        "R² 기준으로는 최근 50년 모델이 테스트 데이터의 기온 변화를 "
        "더 잘 설명했습니다."
    )

elif r2_50 < r2_100:

    st.write(
        "R² 기준으로는 최근 100년 모델이 테스트 데이터의 기온 변화를 "
        "더 잘 설명했습니다."
    )

else:

    st.write(
        "두 모델의 R²가 같습니다."
    )


# =========================================================
# 데이터 상세
# =========================================================

with st.expander("📋 사용된 연평균기온 데이터"):

    st.dataframe(
        yearly[
            [
                "연도",
                "관측일수",
                "연평균기온",
                "지난연수"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


with st.expander("📋 테스트 데이터의 실제값과 예측값"):

    st.dataframe(
        test_result,
        use_container_width=True,
        hide_index=True
    )
