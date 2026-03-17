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

# 1. ŁADOWANIE KONFIGURACJI
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

    # Zmienne Azure (pobierane z Portalu Azure)
    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    AZURE_STORAGE_CONTAINER_NAME: str = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "").strip()
    MODEL_NAME: str = os.getenv("MODEL_NAME", "insurance").strip()
    AZURE_MODEL_BLOB_NAME: str = os.getenv("AZURE_MODEL_BLOB_NAME", "insurance.pkl").strip()

    # Zmienne OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# 2. STRUKTURY DANYCH
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

# 3. LOGIKA AI I MODELU
@st.cache_resource
def load_pipeline(_config: AppConfig) -> Any:
    local_model_name = _config.MODEL_NAME
    local_file_path = f"{local_model_name}.pkl"
    
    try:
        # Pobieranie z Azure Blob Storage jeśli nie ma pliku lokalnie
        if not os.path.exists(local_file_path):
            if not _config.AZURE_STORAGE_CONNECTION_STRING:
                raise RuntimeError("Brak AZURE_STORAGE_CONNECTION_STRING")
            
            blob_service = BlobServiceClient.from_connection_string(_config.AZURE_STORAGE_CONNECTION_STRING)
            blob_client = blob_service.get_blob_client(
                container=_config.AZURE_STORAGE_CONTAINER_NAME,
                blob=_config.AZURE_MODEL_BLOB_NAME
            )
            with open(local_file_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
        
        return load_model(local_model_name)
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {e}")
        st.stop()

def generate_ai_insight(u: UserProfile, active_recos: List[Recommendation], config: AppConfig) -> str:
    if not config.OPENAI_API_KEY:
        return "Błąd: Brak klucza OpenAI w konfiguracji."

    reco_titles = [r.title for r in active_recos] or ["Brak specjalnych rekomendacji."]
    prompt = (
        "Jesteś ekspertem ubezpieczeniowym. Podaj zwięzłą poradę w języku polskim (3 punkty): "
        "(1) główne czynniki ryzyka, (2) kroki na 30 dni, (3) jak obniżyć składkę.\n\n"
        f"Profil: {u.age} lat, BMI: {u.bmi}, Palacz: {u.smoker}\n"
        f"Zalecenia: {', '.join(reco_titles)}"
    )

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Pomocny asystent ubezpieczeniowy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Błąd AI: {e}"

# 4. SILNIK REKOMENDACJI
class RecommendationEngine:
    def __init__(self, config: AppConfig):
        self._config = config
        self._recommendations = self._initialize_recommendations()

    def _initialize_recommendations(self) -> List[Recommendation]:
        return [
            Recommendation(
                id="quit_smoking", title="Rzuć palenie",
                description="Palenie to największy czynnik ryzyka.",
                health_impact="Dramatycznie obniża ryzyko chorób serca.",
                applies_when=lambda u: u.smoker, 
                simulate_change=lambda u: replace(u, smoker=False)
            ),
            Recommendation(
                id="improve_bmi", title="Zredukuj BMI",
                description=f"Celuj w BMI poniżej {self._config.TARGET_BMI}.",
                health_impact="Zapobiega cukrzycy i nadciśnieniu.",
                applies_when=lambda u: u.bmi >= 25.0, 
                simulate_change=lambda u: replace(u, weight_kg=round(self._config.TARGET_BMI * ((u.height_cm/100)**2), 1))
            )
        ]

    def get_for_user(self, u: UserProfile) -> List[Recommendation]:
        return [r for r in self._recommendations if r.applies_when(u)]

# 5. INTERFEJS UŻYTKOWNIKA (UI)
def calculate_final_premium(u: UserProfile, pipeline: Any, config: AppConfig) -> float:
    pred_df = predict_model(pipeline, data=u.to_prediction_input())
    val = float(pred_df['prediction_label'].iloc[0])
    
    # Przeliczenia (USD -> PLN i marża)
    val_pln = val * config.MARKET_ADJUSTMENT_FACTOR * config.USD_TO_PLN_RATE
    monthly = (val_pln / 0.75) / config.MONTHS_IN_YEAR
    if u.prefers_higher_deductible: monthly *= 0.85
    if u.has_group_option: monthly *= config.GROUP_POLICY_DISCOUNT
    
    return round(monthly, 2)

def main():
    config = AppConfig()
    st.set_page_config(page_title=config.PAGE_TITLE, layout="wide")
    
    # Sidebar
    st.sidebar.header("Twoje Dane")
    age = st.sidebar.number_input("Wiek", 18, 100, 30)
    sex = st.sidebar.selectbox("Płeć", ["male", "female"])
    h = st.sidebar.number_input("Wzrost (cm)", 120, 220, 180)
    w = st.sidebar.number_input("Waga (kg)", 40, 200, 80)
    smoker = st.sidebar.toggle("Palacz?")
    children = st.sidebar.number_input("Liczba dzieci", 0, 10, 0)
    act = st.sidebar.slider("Aktywność (dni/tydz)", 0, 7, 2)
    alc = st.sidebar.slider("Alkohol (jednostki/tydz)", 0, 20, 2)
    reg = st.sidebar.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"])
    group = st.sidebar.toggle("Opcja grupowa?", True)
    deduct = st.sidebar.toggle("Wyższy udział własny?", False)

    user = UserProfile(age, sex, h, w, smoker, children, act, alc, [], reg, group, deduct)
    
    # Stan aplikacji
    if 'simulations' not in st.session_state: st.session_state.simulations = {}
    
    pipeline = load_pipeline(config)
    reco_engine = RecommendationEngine(config)
    base_p = calculate_final_premium(user, pipeline, config)

    # Dashboard
    st.title(config.PAGE_TITLE)
    c1, c2, c3 = st.columns(3)
    c1.metric("Twoje BMI", user.bmi)
    c2.metric("Składka (Miesięcznie)", f"{base_p} PLN")
    c3.metric("Status", "Palacz" if user.smoker else "Niepalący")

    st.divider()
    
    # Rekomendacje
    st.subheader("Rekomendacje i Symulacje")
    active_recos = reco_engine.get_for_user(user)
    
    for r in active_recos:
        with st.expander(r.title):
            st.write(r.description)
            if st.button(f"Symuluj oszczędność: {r.title}", key=r.id):
                mod_user = r.simulate_change(user)
                new_p = calculate_final_premium(mod_user, pipeline, config)
                st.session_state.simulations[r.id] = base_p - new_p
            
            if r.id in st.session_state.simulations:
                st.success(f"Możliwa oszczędność: {st.session_state.simulations[r.id]:.2f} PLN / mies")

    st.divider()
    
    # AI Insight
    if st.button("Generuj poradę AI"):
        with st.spinner("AI myśli..."):
            insight = generate_ai_insight(user, active_recos, config)
            st.info(insight)

if __name__ == "__main__":
    main()