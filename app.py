import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="년도별 데이터 입력 & 예측", layout="wide")

st.title("e-지방지표 데이터를 선형 회귀 모델로 예측하기")

# ----------------------------
# 1) 컬럼 이름 설정
# ----------------------------
with st.expander("① 속성 이름 설정 (필수)", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        col_year = st.text_input("년도", value="년도")
    with c2:
        col_target = st.text_input("예측값(target)", value="종속변수")
    with c3:
        col_f1 = st.text_input("예측에 필요한 데이터1(feature)", value="독립변수1")
    with c4:
        col_f2 = st.text_input("예측에 필요한 데이터2(feature)", value="독립변수2")
st.caption("속성 이름을 설정해준 후 데이터를 입력해주세요.")

# ----------------------------
# 2) 6년치 데이터 입력
# ----------------------------
st.markdown("### ② 6년치 데이터 입력")
st.caption("년도, 예측값, 예측에 필요한 데이터1·2를 각각 6행 입력하세요. (숫자만)")

# 기본 6년 템플릿 생성 (2016~2021)
current_year = 2021
base_years = list(range(current_year-5, current_year+1))  # 2016~2021
template = pd.DataFrame({
    col_year: base_years,
    col_target: [np.nan]*6,
    col_f1: [np.nan]*6,
    col_f2: [np.nan]*6,
})

df_input = st.data_editor(
    template,
    num_rows="fixed",
    use_container_width=True,
    key="editor",
)

# ----------------------------
# 3) 미리보기 표 & (정규화) 선 그래프
# ----------------------------
st.markdown("### ③ 그래프로 연관성 확인하기 (정규화 후)")

def validate_df(df: pd.DataFrame):
    required_cols = [col_year, col_target, col_f1, col_f2]
    for c in required_cols:
        if c not in df.columns:
            return False, f"필수 컬럼이 없습니다: {c}"
    try:
        df[col_year] = pd.to_numeric(df[col_year], errors="coerce").astype("Int64")
        df[col_target] = pd.to_numeric(df[col_target], errors="coerce")
        df[col_f1] = pd.to_numeric(df[col_f1], errors="coerce")
        df[col_f2] = pd.to_numeric(df[col_f2], errors="coerce")
    except Exception as e:
        return False, f"숫자 데이터만 입력해주세요: {e}"

    if df[[col_year, col_target, col_f1, col_f2]].isna().any().any():
        return False, "빈 칸(결측치)이 있습니다. 모든 칸을 채워주세요."
    if df[col_year].nunique() < 3:
        return False, "서로 다른 년도가 최소 3개 이상이어야 합니다."
    if len(df) != 6:
        return False, "6행에 데이터를 모두 입력해주세요."
    return True, "OK"

valid, msg = validate_df(df_input.copy())
if not valid:
    st.warning(msg)
else:
    df_clean = df_input.sort_values(col_year).reset_index(drop=True)

    # ---- (추가) 그래프용 표준화: 각 열(종속변수, 독립변수1, 독립변수2)을 z-score로 스케일 ----
    plot_cols = [col_target, col_f1, col_f2]
    df_plot_scaled = df_clean[[col_year] + plot_cols].copy()

    # 각 열 개별 표준화 (평균 0, 표준편차 1) - 그래프 표시 목적
    for c in plot_cols:
        mu = float(df_plot_scaled[c].mean())
        sigma = float(df_plot_scaled[c].std(ddof=0))
        if sigma == 0:
            # 표준편차 0이면 동일값이므로 0으로 설정
            df_plot_scaled[c] = 0.0
        else:
            df_plot_scaled[c] = (df_plot_scaled[c] - mu) / sigma

    # 선 그래프 (Altair) - 정규화된 값으로 표시
    import altair as alt
    plot_df = df_plot_scaled.melt(
        id_vars=[col_year],
        value_vars=plot_cols,
        var_name="변수",
        value_name="표준화값(z-score)"
    )
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{col_year}:O", title="년도"),
            y=alt.Y("표준화값(z-score):Q"),
            color=alt.Color("변수:N"),
            tooltip=[col_year, "변수", "표준화값(z-score)"]
        )
        .properties(width="container", height=360, title="정규화(표준화) 후 선 그래프")
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("세 변수의 스케일 차이를 없애기 위해 각 열을 평균 0, 표준편차 1로 표준화하여 표시했습니다.")

# ----------------------------
# 4) 학습(선형 회귀) & 오차율
# ----------------------------
st.markdown("### ④ 선형 회귀 모델 학습 & 오차율")

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.last_year = None
    st.session_state.feature_cols = [col_f1, col_f2]
    st.session_state.target_col = col_target
    st.session_state.year_col = col_year
    st.session_state.train_mape = None

col_btn1, col_btn2 = st.columns([1, 2])
with col_btn1:
    train_clicked = st.button("학습 완료 & 오차율 보기", use_container_width=True)
with col_btn2:
    st.caption("버튼을 누르면 6년치 데이터를 **표준화 → 선형회귀**로 학습하고, MAPE(평균 절대 백분율 오차)를 표시합니다.")

if train_clicked:
    if not valid:
        st.error("데이터가 올바르지 않습니다. 위 경고를 확인하고 수정하세요.")
    else:
        df_train = df_input.sort_values(col_year).reset_index(drop=True).copy()
        X = df_train[[col_f1, col_f2]].values
        y = df_train[col_target].values

        # ---- (변경) 표준화 + 선형회귀 파이프라인 ----
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(X, y)

        # 학습 세트에서의 오차율(MAPE)
        y_pred = model.predict(X)
        mape = mean_absolute_percentage_error(y, y_pred) * 100  # %
        st.session_state.model_trained = True
        st.session_state.model = model
        st.session_state.last_year = int(df_train[col_year].max())
        st.session_state.train_mape = mape
        st.session_state.feature_cols = [col_f1, col_f2]
        st.session_state.target_col = col_target
        st.session_state.year_col = col_year

        st.success("✅ 학습 완료! 표준화 + 선형 회귀로 학습했습니다.")
        st.info(f"오차율(MAPE): **{mape:.2f}%**")

        # --- 회귀식 표시 (파이프라인의 마지막 단계에서 추출) ---
        lr = model.named_steps["linearregression"] if hasattr(model, "named_steps") else model
        coef = lr.coef_
        intercept = lr.intercept_

        feature_names = [col_f1, col_f2]  # 사용한 특성명 순서와 X의 열 순서가 일치해야 함
        terms = [f"{coef[i]:+.4f}×{feature_names[i]}" for i in range(len(feature_names))]
        equation_text = f"{col_target} = {intercept:.4f} " + " ".join(terms)

        st.markdown("#### 회귀식")
        st.code(equation_text, language="text")

        # LaTeX 버전(옵션)
        # def to_latex_name(name: str) -> str:
        #     return r"\mathrm{" + name.replace("}", r"\}") \
        #                      .replace("{", r"\{") \
        #                      .replace("_", r"\_") + "}"
        # lhs = to_latex_name(col_target)
        # rhs_terms = [f"{coef[i]:+.4f}\\times {to_latex_name(feature_names[i])}" for i in range(len(feature_names))]
        # latex_eq = rf"\hat{{{lhs}}} = {intercept:.4f} " + " ".join(rhs_terms)
        # st.latex(latex_eq)

# ----------------------------
# 5) +1, +2년 예측 (표 입력 및 결과)
# ----------------------------
st.markdown("### ⑤ +1, +2년 예측 (표로 입력)")

if st.session_state.model_trained and st.session_state.model is not None:
    ycol = st.session_state.target_col
    f1, f2 = st.session_state.feature_cols
    last_year = st.session_state.last_year

    st.caption(f"마지막 입력 연도: {last_year} → 예측 대상: {last_year+1}, {last_year+2}")

    # 데이터 입력폼을 가로로 배치
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            f1_y1 = st.number_input(f"{last_year+1}년 {f1}", value=0.0, step=1.0)
            f1_y2 = st.number_input(f"{last_year+2}년 {f1}", value=0.0, step=1.0)
        
        with c2:
            f2_y1 = st.number_input(f"{last_year+1}년 {f2}", value=0.0, step=1.0)
            f2_y2 = st.number_input(f"{last_year+2}년 {f2}", value=0.0, step=1.0)
        
        submitted = st.form_submit_button("예측하기")

    if submitted:
        # 표준화된 입력값으로 변환
        scaler = st.session_state.model.named_steps["standardscaler"]
        X_new = np.array([[f1_y1, f2_y1], [f1_y2, f2_y2]])
        X_new_scaled = scaler.transform(X_new)  # 표준화된 값으로 변환

        # 예측
        preds = st.session_state.model.predict(X_new_scaled)

        # 예측 결과 출력
        result_df = pd.DataFrame({
            st.session_state.year_col: [last_year+1, last_year+2],
            ycol: preds.round(4),
            f1: [f1_y1, f1_y2],
            f2: [f2_y1, f2_y2],
        })

        st.success("예측 완료!")
        st.write(f"학습 시 오차율(MAPE): **{st.session_state.train_mape:.2f}%**")
        st.dataframe(result_df, use_container_width=True)

        # 예측 포함 그래프(타깃만) - 원값 기준
        import altair as alt
        if valid:
            future_rows = result_df[[st.session_state.year_col, "예측값"]].copy()
            future_rows["데이터구분"] = "예측"

            hist_rows = df_input[[st.session_state.year_col, ycol]].copy()
            hist_rows["데이터구분"] = "실제"

            plot2 = pd.concat([hist_rows, future_rows], ignore_index=True)
            plot2 = plot2.rename(columns={st.session_state.year_col: "년도", "예측값": "예측값"})
            chart2 = (
                alt.Chart(plot2)
                .mark_line(point=True)
                .encode(
                    x=alt.X("년도:O", title="년도"),
                    y=alt.Y("예측값:Q"),
                    color=alt.Color("데이터구분:N"),
                    tooltip=["년도", "예측값", "데이터구분"]
                )
                .properties(width="container", height=320, title="예측값(타깃) 추세")
            )
            st.altair_chart(chart2, use_container_width=True)

else:
    st.info("먼저 위에서 **학습 완료 & 오차율 보기** 버튼을 눌러 모델을 학습하세요.")
