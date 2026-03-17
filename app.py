import os
from dataclasses import dataclass, replace
from typing import List, Any, Callable, Optional
import pandas as pd
import streamlit as st
import plotly.express as px
from azure.storage.blob import BlobServiceClient
from openai import OpenAI
from pycaret.regression import load_model, predict_model
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    PAGE_TITLE: str = "AI Insurance Premium Advisor"
    USD_TO_PLN_RATE: float = 4.0
    MONTHS_IN_YEAR: int = 12
    TARGET_BMI: float = 24.9
    MARKET_ADJUSTMENT_FACTOR: float = 0.2
    GROUP_POLICY_DISCOUNT: float = 0.85
    ALCOHOL_UNITS_THRESHOLD: int = 7
    ACTIVITY_DAYS_THRESHOLD: int = 3

    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    AZURE_STORAGE_CONTAINER_NAME: str = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "").strip()
    MODEL_NAME: str = os.getenv("MODEL_NAME", "insurance").strip()
    AZURE_MODEL_BLOB_NAME: str = os.getenv("AZURE_MODEL_BLOB_NAME", "").strip()

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

@dataclass(frozen=True)
class UserProfile:
    age: int; sex: str; height_cm: float; weight_kg: float; smoker: bool
    children: int; weekly_activity_days: int; alcohol_units_week: int
    conditions: List[str]; region: str; has_group_option: bool
    prefers_higher_deductible: bool

    @property
    def bmi(self) -> float:
        height_m = max(self.height_cm / 100.0, 0.5)
        return round(self.weight_kg / (height_m ** 2), 1)

    def to_prediction_input(self) -> pd.DataFrame:
        return pd.DataFrame({
            'age': [self.age], 'sex': [self.sex], 'bmi': [self.bmi],
            'children': [self.children], 'smoker': ['yes' if self.smoker else 'no'],
            'region': [self.region], 'weekly_activity_days': [self.weekly_activity_days],
            'alcohol_units_week': [self.alcohol_units_week],
            'has_conditions': [1 if self.conditions else 0]
        })

@dataclass(frozen=True)
class Recommendation:
    id: str; title: str; description: str
    health_impact: Optional[str]
    applies_when: Callable[[UserProfile], bool]
    simulate_change: Callable[[UserProfile], UserProfile]

@dataclass(frozen=True)
class AppState:
    profile: UserProfile; pipeline: Any; engine: Any; config: AppConfig
    base_premium: float; multiplier: int; period_label: str

@st.cache_resource
def load_pipeline(_config: AppConfig) -> Any:
    local_model_name = _config.MODEL_NAME
    local_file_path = f"{local_model_name}.pkl"
    blob_name = _config.AZURE_MODEL_BLOB_NAME or local_file_path
    
    try:
        if not os.path.exists(local_file_path):
            if not _config.AZURE_STORAGE_CONNECTION_STRING:
                raise RuntimeError("Missing AZURE_STORAGE_CONNECTION_STRING in environment.")
            if not _config.AZURE_STORAGE_CONTAINER_NAME:
                raise RuntimeError("Missing AZURE_STORAGE_CONTAINER_NAME in environment.")

            blob_service = BlobServiceClient.from_connection_string(_config.AZURE_STORAGE_CONNECTION_STRING)
            blob_client = blob_service.get_blob_client(
                container=_config.AZURE_STORAGE_CONTAINER_NAME,
                blob=blob_name,
            )
            with open(local_file_path, "wb") as model_file:
                model_file.write(blob_client.download_blob().readall())
            
        return load_model(local_model_name, verbose=False)
        
    except Exception as e:
        st.error(f"Critical System Error: {e}")
        st.stop()

def _calculate_base_premium(u: UserProfile, pipeline: Any, config: AppConfig) -> float:
    input_df = u.to_prediction_input()
    pred_df = predict_model(pipeline, data=input_df)
    
    expected_usd_year = float(pred_df['prediction_label'].iloc[0])
    adjusted_usd_year = expected_usd_year * config.MARKET_ADJUSTMENT_FACTOR

    expected_loss_pln_year = adjusted_usd_year * config.USD_TO_PLN_RATE
    gross_premium_year = (expected_loss_pln_year / 0.75) * (0.85 if u.prefers_higher_deductible else 1.0)
    
    return gross_premium_year / config.MONTHS_IN_YEAR

def calculate_final_premium(u: UserProfile, pipeline: Any, config: AppConfig) -> float:
    base_premium = _calculate_base_premium(u, pipeline, config)
    final_premium = base_premium * config.GROUP_POLICY_DISCOUNT if u.has_group_option else base_premium
    return round(final_premium, 2)

def _build_profile_summary(u: UserProfile) -> str:
    return (
        f"Age: {u.age}, Sex: {u.sex}, BMI: {u.bmi}, Smoker: {u.smoker}, "
        f"Children: {u.children}, Activity days/week: {u.weekly_activity_days}, "
        f"Alcohol units/week: {u.alcohol_units_week}, Conditions: {', '.join(u.conditions) if u.conditions else 'none'}, "
        f"Region: {u.region}, Group policy option: {u.has_group_option}, "
        f"Prefers higher deductible: {u.prefers_higher_deductible}"
    )

def generate_ai_insight(u: UserProfile, active_recos: List[Recommendation], config: AppConfig) -> str:
    if not config.OPENAI_API_KEY:
        return "Brak OPENAI_API_KEY w .env."

    reco_titles = [r.title for r in active_recos] or ["No active recommendations"]
    prompt = (
        "You are a health-insurance assistant. "
        "Provide a concise recommendation in Polish with 3 bullet points: "
        "(1) main risk factors, (2) practical next steps for 30 days, "
        "(3) what can lower insurance premium the most. "
        "Keep response under 120 words and avoid medical diagnosis.\n\n"
        f"User profile: {_build_profile_summary(u)}\n"
        f"Current app recommendations: {', '.join(reco_titles)}"
    )

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.responses.create(
            model=config.OPENAI_MODEL,
            input=prompt,
            temperature=0.4,
            max_output_tokens=220,
        )
        return (response.output_text or "Nie udało się wygenerować porady AI.").strip()
    except Exception as exc:
        return f"Błąd OpenAI: {exc}"

class RecommendationEngine:
    def __init__(self, config: AppConfig):
        self._config = config
        self._recommendations = self._initialize_recommendations()

    def _get_target_weight(self, height_cm: float) -> float:
        h_m = height_cm / 100.0
        return round(self._config.TARGET_BMI * (h_m ** 2), 1)

    def _initialize_recommendations(self) -> List[Recommendation]:
        return [
            Recommendation(
                id="quit_smoking", title="Quit Smoking",
                description="The single largest risk factor. Quitting yields the highest financial and health benefits.",
                health_impact="Smoking drastically increases the risk of heart disease, cancer, and chronic respiratory issues. Quitting is the most critical step for a longer life.",
                applies_when=lambda u: u.smoker, 
                simulate_change=lambda u: replace(u, smoker=False)
            ),
            Recommendation(
                id="improve_bmi", title=f"Reduce BMI to normal (< {self._config.TARGET_BMI})",
                description="Reaching a healthy weight significantly lowers the risk of chronic diseases, which reduces your premium.",
                health_impact="Overweight and obesity (BMI >= 25) lead to hypertension, type 2 diabetes, and heart disease. Maintaining a healthy weight is fundamental for preventive care.",
                applies_when=lambda u: u.bmi >= 25.0, 
                simulate_change=lambda u: replace(u, weight_kg=self._get_target_weight(u.height_cm))
            ),
            Recommendation(
                id="increase_activity", title="Increase Physical Activity",
                description=f"Increasing activity to at least {self._config.ACTIVITY_DAYS_THRESHOLD} days a week is key to better health and lower premiums.",
                health_impact="Low physical activity is a primary risk factor for lifestyle diseases. Regular movement regulates blood pressure and lowers LDL cholesterol.",
                applies_when=lambda u: u.weekly_activity_days < self._config.ACTIVITY_DAYS_THRESHOLD, 
                simulate_change=lambda u: replace(u, weekly_activity_days=self._config.ACTIVITY_DAYS_THRESHOLD)
            ),
            Recommendation(
                id="reduce_alcohol", title="Reduce Alcohol Consumption",
                description=f"Limiting consumption to a maximum of {self._config.ALCOHOL_UNITS_THRESHOLD} units per week improves your risk profile.",
                health_impact=f"Consuming more than {self._config.ALCOHOL_UNITS_THRESHOLD} units of alcohol weekly heavily burdens the liver and increases the risk of heart disease.",
                applies_when=lambda u: u.alcohol_units_week > self._config.ALCOHOL_UNITS_THRESHOLD, 
                simulate_change=lambda u: replace(u, alcohol_units_week=self._config.ALCOHOL_UNITS_THRESHOLD)
            ),
            Recommendation(
                id="group_policy_benefit", title="Check Group Policy Benefit",
                description="See how much you save with this option compared to a standard individual offer.",
                health_impact=None,
                applies_when=lambda u: u.has_group_option, 
                simulate_change=lambda u: replace(u, has_group_option=False)
            ),
        ]

    def get_for_user(self, user_profile: UserProfile) -> List[Recommendation]:
        return [r for r in self._recommendations if r.applies_when(user_profile)]

def ui_sidebar(config: AppConfig) -> UserProfile:
    st.sidebar.header("Enter your details")
    with st.sidebar:
        age = st.number_input("Age", 18, 100, 30, key="age")
        sex_map = {"Female": "female", "Male": "male"}
        sex_display = st.selectbox("Gender", list(sex_map.keys()), index=1, key="sex")
        height_cm = st.number_input("Height [cm]", 120, 220, 180, key="height")
        weight_kg = st.number_input("Weight [kg]", 40, 250, 85, key="weight")
        st.divider()
        smoker = st.toggle("Do you smoke?", False, key="smoker")
        children = st.number_input("Number of children", 0, 10, 0, key="children")
        weekly_activity_days = st.slider("Active days per week", 0, 7, 1, key="activity")
        alcohol_units_week = st.slider("Alcohol units per week", 0, 7, 5, key="alcohol")
        st.divider()
        conditions = st.multiselect("Chronic conditions", ["hypertension", "diabetes"], key="conditions")
        
        region_map = {
            "West Pomeranian": "northwest", "Pomeranian": "northwest", "Kuyavian-Pomeranian": "northwest",
            "Greater Poland": "northwest", "Lubusz": "northwest", "Warmian-Masurian": "northeast",
            "Podlaskie": "northeast", "Masovian": "northeast", "Lower Silesian": "southwest",
            "Opole": "southwest", "Silesian": "southwest", "Łódź": "southeast", "Świętokrzyskie": "southeast",
            "Lublin": "southeast", "Subcarpathian": "southeast", "Lesser Poland": "southeast"
        }
        region_display = st.selectbox("Region", list(region_map.keys()), index=1, key="region")
        st.divider()
        has_group_option = st.toggle("Do you have a group policy option?", True, key="group_option", help="Policy offered by an employer, usually on better terms.")
        prefers_higher_deductible = st.toggle("Considering a higher deductible?", False, key="deductible", help="Means a lower premium in exchange for taking on a larger share of potential claim costs.")

        return UserProfile(
            age=age, sex=sex_map[sex_display], height_cm=height_cm, weight_kg=weight_kg, smoker=smoker,
            children=children, weekly_activity_days=weekly_activity_days, alcohol_units_week=alcohol_units_week,
            conditions=conditions, region=region_map[region_display], has_group_option=has_group_option,
            prefers_higher_deductible=prefers_higher_deductible
        )

def ui_dashboard(state: AppState):
    st.subheader("Your Personalized Analysis")
    k1, k2, k3 = st.columns(3)

    with k1:
        bmi = state.profile.bmi
        color, status = ("green", "Normal") if 18.5 <= bmi < 25 else ("red", "Out of range")
        st.markdown(f"""
        <div style="line-height: 1.2; height: 100%;"><p style="font-size: 0.9rem; color: #808495; margin-bottom: 0;">Your BMI</p><p style="font-size: 1.75rem; font-weight: 600; margin-bottom: 0;">{bmi}</p><p style="color: {color}; margin-bottom: 0;">{status}</p></div>
        """, unsafe_allow_html=True)

    k2.metric("Estimated Premium", f"{state.base_premium * state.multiplier:.2f} PLN{state.period_label}")
    k3.metric("Smoking Status", "Smoker" if state.profile.smoker else "Non-smoker")

def ui_recommendations(state: AppState):
    st.subheader("How to lower your premium and improve your health")
    st.caption("Click the button to see a precise savings simulation.")

    active_recos = state.engine.get_for_user(state.profile)
    if not active_recos:
        st.success("Your profile is optimal. No further recommendations available.")
        return

    for reco in active_recos:
        with st.expander(f"{reco.title}"):
            st.write(reco.description)
            
            if reco.health_impact:
                st.info(f"Health Impact: {reco.health_impact}")

            if st.button(f"Simulate: {reco.title}", key=f"btn_{reco.id}"):
                modified_profile = reco.simulate_change(state.profile)
                new_premium = calculate_final_premium(modified_profile, state.pipeline, state.config)
                savings = (state.base_premium - new_premium) if reco.id != "group_policy_benefit" else (new_premium - state.base_premium)
                st.session_state.simulations[reco.id] = {"new_premium": new_premium, "savings": savings}
            
            if reco.id in st.session_state.simulations:
                sim = st.session_state.simulations[reco.id]
                sav = sim['savings'] * state.multiplier
                if sav > 0.01:
                    msg = (f"Savings with this option: {sav:.2f} PLN{state.period_label}" if reco.id == "group_policy_benefit" else f"New premium: {sim['new_premium'] * state.multiplier:.2f} PLN{state.period_label} | Savings: {sav:.2f} PLN{state.period_label}")
                    st.success(f"{msg}")
                else:
                    st.info("This simulation shows no savings for your current profile.")

    st.divider()
    st.markdown("### AI Insight (OpenAI)")
    if st.button("Generate AI insight", key="btn_ai_insight"):
        with st.spinner("Generating AI insight..."):
            st.session_state.ai_insight = generate_ai_insight(state.profile, active_recos, state.config)

    if st.session_state.get("ai_insight"):
        st.info(st.session_state.ai_insight)

def ui_savings_chart(state: AppState):
    if not st.session_state.get('simulations'):
        return
        
    st.subheader("Savings Visualization")
    
    all_recos_map = {r.id: r.title for r in state.engine._recommendations}
    plot_data = []
    
    for reco_id, sim_data in st.session_state.simulations.items():
        if sim_data['savings'] > 0.01:
            reco_title = all_recos_map.get(reco_id, "Unknown recommendation")
            base_premium_adj = state.base_premium * state.multiplier
            new_premium_adj = sim_data['new_premium'] * state.multiplier

            plot_data.extend([
                {'Recommendation': reco_title, 'Premium': base_premium_adj, 'Type': 'Current Premium'},
                {'Recommendation': reco_title, 'Premium': new_premium_adj, 'Type': 'Premium after change'}
            ])

    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        fig = px.bar(df_plot, x="Recommendation", y="Premium", color="Type", barmode='group', title="Premium Comparison", labels={"Premium": f"Premium [PLN{state.period_label}]", "Recommendation": "", "Type": "Premium Type"}, text_auto='.2f')
        fig.update_traces(textangle=0, textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

def manage_session_state(current_profile: UserProfile):
    if 'simulations' not in st.session_state or st.session_state.get('last_profile') != current_profile:
        st.session_state.simulations = {}
        st.session_state.last_profile = current_profile

def main():
    config = AppConfig()
    st.set_page_config(page_title=config.PAGE_TITLE, layout="wide")
    st.title(config.PAGE_TITLE)
    
    pipeline = load_pipeline(config)
    reco_engine = RecommendationEngine(config)
    user_profile = ui_sidebar(config)

    manage_session_state(user_profile)

    base_premium = calculate_final_premium(user_profile, pipeline, config)
    
    view = st.radio("Show costs:", ["Monthly", "Annually"], horizontal=True, index=0)
    multiplier = config.MONTHS_IN_YEAR if view == "Annually" else 1
    period_label = "/yr" if view == "Annually" else "/mo"
    
    app_state = AppState(
        profile=user_profile, pipeline=pipeline, engine=reco_engine, config=config, 
        base_premium=base_premium, multiplier=multiplier, 
        period_label=period_label
    )
    
    st.divider()
    ui_dashboard(app_state)
    st.divider()
    ui_recommendations(app_state)
    st.divider()
    ui_savings_chart(app_state)

if __name__ == "__main__":
    main()