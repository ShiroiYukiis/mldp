import importlib
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


def _load_joblib_module():
    """Load joblib when available and fall back to pickle otherwise."""
    try:
        return importlib.import_module("joblib")
    except ModuleNotFoundError:
        class _FallbackJoblib:
            @staticmethod
            def load(path):
                with open(path, "rb") as handle:
                    return pickle.load(handle)

            @staticmethod
            def dump(value, path):
                with open(path, "wb") as handle:
                    pickle.dump(value, handle)

        return _FallbackJoblib()


joblib = _load_joblib_module()


APP_TITLE = "Sleep Pattern Checker"
MODEL_PATH = Path(__file__).resolve().parent / "sleep_disorder_model.joblib"

FEATURE_LABELS = {
    "Gender": "Gender",
    "Age": "Age",
    "Occupation": "Occupation",
    "Sleep Duration": "Sleep duration per night",
    "Quality of Sleep": "Sleep quality",
    "Physical Activity Level": "Daily physical activity",
    "BMI Category": "BMI category",
    "Systolic BP": "Upper blood-pressure reading",
    "Diastolic BP": "Lower blood-pressure reading",
    "Heart Rate": "Resting heart rate",
    "Daily Steps": "Average daily steps",
    "Stress Level": "Stress level",
}

FORM_WIDGET_KEYS = {
    "Gender": "gender_input",
    "Age": "age_input",
    "Occupation": "occupation_input",
    "Sleep Duration": "sleep_duration_input",
    "Quality of Sleep": "sleep_quality_input",
    "Physical Activity Level": "physical_activity_input",
    "Stress Level": "stress_level_input",
    "BMI Category": "bmi_category_input",
    "Systolic BP": "systolic_bp_input",
    "Diastolic BP": "diastolic_bp_input",
    "Heart Rate": "heart_rate_input",
    "Daily Steps": "daily_steps_input",
}


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded", 
)


st.markdown(
    """
    <style>
        :root {
            color-scheme: light dark;
        }

        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background-color: var(--background-color);
            border-right: 1px solid
                color-mix(
                    in srgb,
                    var(--text-color) 16%,
                    transparent
                );
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p {
            color: var(--text-color);
        }

        [data-testid="stWidgetLabel"] p {
            font-weight: 600;
        }

        div[data-testid="stForm"] {
            background-color: var(--secondary-background-color);
            border: 1px solid
                color-mix(
                    in srgb,
                    var(--text-color) 16%,
                    transparent
                );
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            padding: 1.5rem;
        }

        div[data-testid="stForm"] h3 {
            color: var(--text-color);
            font-size: 1.3rem;
            margin-bottom: 1rem;
        }

        [data-testid="stFormSubmitButton"] button {
            min-height: 3rem;
            border-radius: 9px;
            font-weight: 700;
        }

        .result-card {
            background-color: var(--secondary-background-color);
            border: 1px solid
                color-mix(
                    in srgb,
                    var(--text-color) 14%,
                    transparent
                );
            border-left: 6px solid var(--primary-color);
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
            margin: 1rem 0;
            padding: 1.25rem 1.5rem;
        }

        .result-label {
            color: var(--text-color) !important;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
            opacity: 0.75;
        }

        .result-value {
            color: var(--text-color) !important;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }

        .small-note {
            color: var(--text-color) !important;
            font-size: 0.9rem;
            opacity: 0.78;
        }
        .stApp h1 {
            color: var(--primary-color);
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stDownloadButton"] button {
            min-height: 3rem;
            border-radius: 999px;
            font-weight: 700;
        }

        div[data-testid="stExpander"] {
            border-radius: 14px;
            overflow: hidden;
        }

        .result-card {
            border-left-color: #14B8A6;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_package(model_path: Path) -> dict:
    """Load and validate the saved model package."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path.name}"
        )

    package = joblib.load(model_path)

    required_keys = {
        "model",
        "selected_features",
        "categorical_features",
        "category_levels",
        "numerical_input_features",
        "input_ranges",
        "target_classes",
    }

    missing_keys = required_keys.difference(package)
    if missing_keys:
        missing_text = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Model package is missing: {missing_text}"
        )

    if not hasattr(package["model"], "predict_proba"):
        raise ValueError(
            "The saved model does not support probability predictions."
        )

    raw_features = set(package["numerical_input_features"]).union(
        package["categorical_features"]
    )
    unsupported_features = raw_features.difference(FORM_WIDGET_KEYS)
    if unsupported_features:
        unsupported_text = ", ".join(sorted(unsupported_features))
        raise ValueError(
            f"The app does not have input controls for: {unsupported_text}"
        )

    missing_ranges = set(
        package["numerical_input_features"]
    ).difference(package["input_ranges"])
    if missing_ranges:
        raise ValueError(
            "The model package does not contain every numerical input range."
        )

    missing_category_levels = set(
        package["categorical_features"]
    ).difference(package["category_levels"])
    if missing_category_levels:
        raise ValueError(
            "The model package does not contain every category list."
        )

    return package


def default_value(
    input_ranges: dict,
    feature: str,
    preferred_value: float,
) -> float:
    """Keep a default value inside the dataset-supported range."""
    lower = float(input_ranges[feature]["min"])
    upper = float(input_ranges[feature]["max"])
    return min(max(preferred_value, lower), upper)


def choose_category(
    available_values: list,
    preferred_values: list[str],
):
    """Return the first preferred category that exists in the model package."""
    value_lookup = {
        str(value).strip().lower(): value
        for value in available_values
    }

    for preferred_value in preferred_values:
        match = value_lookup.get(preferred_value.strip().lower())
        if match is not None:
            return match

    return available_values[0] if available_values else None


def format_bmi_category(category: str) -> str:
    """Show BMI categories in language that is easier for users to follow."""
    normalised_category = str(category).strip().lower()

    descriptions = {
    "normal": "Healthy BMI range — 18.5 to 24.9",
    "normal weight": "Healthy BMI range — 18.5 to 24.9",
    "healthy weight": "Healthy BMI range — 18.5 to 24.9",
    "overweight": "Overweight BMI range — 25.0 to 29.9",
    "obese": "Obesity BMI range — 30.0 or higher",
    "obesity": "Obesity BMI range — 30.0 or higher",
    "underweight": "Underweight BMI range — below 18.5",
}

    return descriptions.get(normalised_category, str(category))


def build_text_summary(result: dict) -> str:
    """Create a simple downloadable summary without storing personal data."""
    lines = [
        "Your Sleep Pattern Summary",
        "",
        f"Your sleep pattern result: {result['prediction']}",
        f"Match level: {result['highest_probability']:.1f}%",
        "",
        "How closely your answers matched each sleep pattern:",
    ]

    for row in result["probability_table"].to_dict("records"):
        lines.append(
            f"- {row['Class']}: {row['Probability (%)']:.1f}%"
        )

    lines.extend(["", "Information you provided:"])

    for feature, value in result["raw_input"].items():
        label = FEATURE_LABELS.get(feature, feature)
        lines.append(f"- {label}: {value}")

    lines.extend(
        [
            "",
            (
                "Please remember that this result is only a general guide, "
                "not a medical diagnosis. If you are worried about your sleep "
                "or health, speak with a qualified healthcare professional."
            ),
        ]
    )

    return "\n".join(lines)


def validate_input(
    raw_input: dict,
    package: dict,
) -> list[str]:
    """Return clear validation messages for invalid input values."""
    errors = []

    for feature in package["numerical_input_features"]:
        value = raw_input.get(feature)
        limits = package["input_ranges"][feature]
        lower = float(limits["min"])
        upper = float(limits["max"])
        label = FEATURE_LABELS.get(feature, feature)

        if value is None or pd.isna(value):
            errors.append(
                f"Please enter your {label.lower()}."
            )
        elif not lower <= float(value) <= upper:
            errors.append(
                f"Please enter {label.lower()} from {lower:g} to {upper:g}."
            )

    for feature in package["categorical_features"]:
        value = raw_input.get(feature)
        allowed_values = package["category_levels"][feature]
        label = FEATURE_LABELS.get(feature, feature)

        if value not in allowed_values:
            errors.append(
                f"Please choose an option for {label.lower()}."
            )

    systolic_bp = raw_input.get("Systolic BP")
    diastolic_bp = raw_input.get("Diastolic BP")
    if (
        systolic_bp is not None
        and diastolic_bp is not None
        and float(diastolic_bp) >= float(systolic_bp)
    ):
        errors.append(
            "Please check your blood pressure reading. The lower number "
            "should be smaller than the upper number."
        )

    return errors


def prepare_model_input(
    raw_input: dict,
    package: dict,
) -> pd.DataFrame:
    """Apply the same category encoding and column order used in training."""
    input_frame = pd.DataFrame([raw_input])

    for feature in package["categorical_features"]:
        input_frame[feature] = pd.Categorical(
            input_frame[feature],
            categories=package["category_levels"][feature],
        )

    encoded_input = pd.get_dummies(
        input_frame,
        columns=package["categorical_features"],
        drop_first=True,
        dtype=int,
    )

    model_input = encoded_input.reindex(
        columns=package["selected_features"],
        fill_value=0,
    )

    if model_input.isnull().any().any():
        raise ValueError(
            "Some of the entered information could not be processed."
        )

    return model_input


try:
    model_package = load_model_package(MODEL_PATH)

except FileNotFoundError:
    st.error(
        "Sorry, the sleep checker is temporarily unavailable because "
        "a required file is missing. Please try again later."
    )
    st.stop()

except (ValueError, KeyError, TypeError):
    st.error(
        "Sorry, the sleep checker could not start because some required "
        "information could not be read. Please try again later."
    )
    st.stop()

except Exception:
    st.error(
        "Sorry, we could not start the sleep checker. Please refresh "
        "the page or try again later."
    )
    st.stop()


model = model_package["model"]
input_ranges = model_package["input_ranges"]
category_levels = model_package["category_levels"]
numerical_input_features = list(
    model_package["numerical_input_features"]
)
categorical_input_features = list(
    model_package["categorical_features"]
)
active_features = set(numerical_input_features).union(
    categorical_input_features
)


def preferred_numeric_value(feature: str, preferred_value: float):
    """Return an integer or float default that fits the saved input range."""
    value = default_value(input_ranges, feature, preferred_value)
    if feature == "Sleep Duration":
        return float(value)
    return int(value)


def render_number_input(feature: str):
    """Render a numerical field using the model's supported range."""
    labels = {
        "Age": "Your age (years)",
        "Sleep Duration": "Usual sleep per night (hours)",
        "Quality of Sleep": "How well do you usually sleep?",
        "Physical Activity Level": "Daily physical activity (minutes)",
        "Stress Level": "Your usual stress level",
        "Systolic BP": "Upper blood pressure number (mmHg)",
        "Diastolic BP": "Lower blood pressure number (mmHg)",
        "Heart Rate": "Resting heart rate (beats per minute)",
        "Daily Steps": "Usual number of steps per day",
    }

    help_text = {
        "Age": (
            "Enter your age from 27 to 60 years."
        ),
        "Sleep Duration": (
            "Enter how long you usually sleep each night, from 5.8 to 8.5 hours."
        ),
        "Quality of Sleep": (
            "Choose a rating from 4 to 9. A higher number means better sleep."
        ),
        "Physical Activity Level": (
            "Enter how many minutes you are usually active each day, "
            "from 30 to 90 minutes."
        ),
        "Stress Level": (
            "Choose the number that best describes your usual stress. "
            "A higher number means more stress."
        ),
        "Systolic BP": (
            "Enter the upper number from your blood pressure reading, "
            "from 115 to 142 mmHg."
        ),
        "Diastolic BP": (
            "This is the lower number in a blood pressure reading, such as "
            "80 in 120/80."
        ),
        "Heart Rate": (
            "Enter your resting heart rate from 65 to 86 beats per minute."
        ),
        "Daily Steps": (
            "Enter approximately how many steps you usually take each day."
        ),
    }

    lower = input_ranges[feature]["min"]
    upper = input_ranges[feature]["max"]
    widget_key = FORM_WIDGET_KEYS[feature]

    if feature == "Sleep Duration":
        return st.number_input(
            labels[feature],
            min_value=float(lower),
            max_value=float(upper),
            step=0.1,
            format="%.1f",
            key=widget_key,
            help=help_text.get(feature),
        )

    return st.number_input(
        labels[feature],
        min_value=int(lower),
        max_value=int(upper),
        step=1,
        key=widget_key,
        help=help_text.get(feature),
    )


default_preferences = {
    "Age": 40,
    "Sleep Duration": 7.0,
    "Quality of Sleep": 7,
    "Physical Activity Level": 60,
    "Stress Level": 5,
    "Daily Steps": 7000,
    "Systolic BP": 130,
    "Heart Rate": 68,
}
health_features = [
    feature
    for feature in ("Systolic BP", "Diastolic BP", "Heart Rate")
    if feature in numerical_input_features
]

default_form_values = {}

for feature in categorical_input_features:
    default_form_values[feature] = None

for feature in numerical_input_features:
    if feature in health_features and feature not in default_preferences:
        default_form_values[feature] = None
    else:
        default_form_values[feature] = preferred_numeric_value(
            feature,
            default_preferences.get(
                feature,
                input_ranges[feature]["min"],
            ),
        )

example_preferences = {
    "Age": 45,
    "Sleep Duration": 6.5,
    "Quality of Sleep": 6,
    "Physical Activity Level": 45,
    "Stress Level": 6,
    "Systolic BP": 130,
    "Diastolic BP": 85,
    "Heart Rate": 75,
    "Daily Steps": 6000,
}
example_form_values = {
    feature: preferred_numeric_value(
        feature,
        example_preferences.get(feature, input_ranges[feature]["min"]),
    )
    for feature in numerical_input_features
}

category_preferences = {
    "Gender": ["Female", "Male"],
    "Occupation": ["Teacher", "Engineer", "Accountant"],
    "BMI Category": ["Overweight", "Normal", "Normal Weight"],
}
for feature in categorical_input_features:
    example_form_values[feature] = choose_category(
        list(category_levels[feature]),
        category_preferences.get(feature, []),
    )


def apply_form_values(values: dict, notice: str) -> None:
    """Update every active form widget and clear the previous result."""
    for feature, value in values.items():
        if feature in active_features:
            st.session_state[FORM_WIDGET_KEYS[feature]] = value

    st.session_state.pop("screening_result", None)
    st.session_state["form_notice"] = notice


def load_example_values() -> None:
    apply_form_values(
        example_form_values,
        "We've filled in some example answers for you. Feel free to change "
        "them before checking your sleep pattern.",
    )


def reset_form_values() -> None:
    apply_form_values(
        default_form_values,
        "The form has been reset.",
    )
    st.session_state["bmi_height_cm"] = 165.0
    st.session_state["bmi_weight_kg"] = 65.0


for feature, value in default_form_values.items():
    st.session_state.setdefault(
        FORM_WIDGET_KEYS[feature],
        value,
    )
st.session_state.setdefault("bmi_height_cm", 165.0)
st.session_state.setdefault("bmi_weight_kg", 65.0)


with st.sidebar:
    st.header("About this sleep checker")

    st.write(
        "This tool compares your answers with sleep patterns commonly "
        "linked to the following results:"
    )
    st.write("No Sleep Disorder, Insomnia and Sleep Apnea.")

    st.divider()

    with st.expander("How your result is prepared"):
        st.write(
            f"The checker reviews the "
            "details from your answers, including your sleep habits, "
            "daily activities and health readings."
        )
        st.write(
            "It then compares your information with sleep patterns it has "
            "learned and shows the pattern that matches you most closely."
        )

    st.divider()

    st.caption(
        "This tool is for learning and general guidance only. It cannot "
        "diagnose a medical condition or replace advice from a qualified "
        "healthcare professional."
    )


st.title(APP_TITLE)

st.write(
    "Tell us about your sleep, daily habits and health to discover "
    "which sleep pattern most closely matches you."
)

st.warning(
    "This result is a general guide, not a medical diagnosis. "
    "If you are worried about your sleep or health, please speak "
    "with a qualified healthcare professional."
)


action_column, reset_column, spacer_column = st.columns([1.2, 1, 2.8])
with action_column:
    st.button(
        "Try example values",
        on_click=load_example_values,
        use_container_width=True,
    )
with reset_column:
    st.button(
        "Reset form",
        on_click=reset_form_values,
        use_container_width=True,
    )

form_notice = st.session_state.pop("form_notice", None)
if form_notice:
    st.success(form_notice)


if "BMI Category" in categorical_input_features:
    with st.expander("Need help finding your BMI category?"):
        st.write(
            "Enter your height and weight below, and we'll calculate your "
            "BMI and select the matching category for you."
        )

        with st.form("bmi_helper_form"):
            height_column, weight_column = st.columns(2)
            with height_column:
                height_cm = st.number_input(
                    "Height (cm)",
                    min_value=100.0,
                    max_value=250.0,
                    step=0.5,
                    key="bmi_height_cm",
                )
            with weight_column:
                weight_kg = st.number_input(
                    "Weight (kg)",
                    min_value=20.0,
                    max_value=300.0,
                    step=0.5,
                    key="bmi_weight_kg",
                )

            calculate_bmi = st.form_submit_button(
                "Calculate BMI category",
                use_container_width=True,
            )

        if calculate_bmi:
            bmi_value = weight_kg / ((height_cm / 100) ** 2)

            if bmi_value < 18.5:
                bmi_name = "Underweight"
                preferred_names = ["Underweight"]
            elif bmi_value < 25:
                bmi_name = "Normal"
                preferred_names = [
                    "Normal",
                    "Normal Weight",
                    "Healthy Weight",
                ]
            elif bmi_value < 30:
                bmi_name = "Overweight"
                preferred_names = ["Overweight"]
            else:
                bmi_name = "Obese"
                preferred_names = ["Obese", "Obesity"]

            available_bmi_values = list(
                category_levels["BMI Category"]
            )
            matched_category = choose_category(
                available_bmi_values,
                preferred_names,
            )
            matched_text = str(matched_category).strip().lower()
            preferred_text = {
                value.lower()
                for value in preferred_names
            }

            if matched_text in preferred_text:
                st.session_state[
                    FORM_WIDGET_KEYS["BMI Category"]
                ] = matched_category
                st.success(
                    f"Your BMI is {bmi_value:.1f}, which falls within the "
                    f"{bmi_name} range. We've selected this category for you."
                )
            else:
                st.session_state[
                    FORM_WIDGET_KEYS["BMI Category"]
                ] = None
                st.error(
                    f"Your BMI is {bmi_value:.1f}, which falls within the "
                    f"{bmi_name} range. Unfortunately, this category isn't included "
                    "in the sleep checker, so we cannot prepare a reliable result."
                )

        st.caption(
            "BMI is only a general guide and does not provide a complete "
            "picture of your health."
        )


st.caption(
    "BMI is only a general guide and does not provide a complete "
    "picture of your health."
)

with st.form("sleep_screening_form"):
    personal_column, sleep_column = st.columns(2)

    with personal_column:
        st.subheader("About you")

        if "Gender" in categorical_input_features:
            st.selectbox(
                "Gender",
                options=list(category_levels["Gender"]),
                index=None,
                placeholder="Select your gender",
                key=FORM_WIDGET_KEYS["Gender"],
            )

        if "Age" in numerical_input_features:
            render_number_input("Age")
            st.caption("Please enter your age from 27 to 60 years.")

        if "Occupation" in categorical_input_features:
            st.selectbox(
                "Occupation",
                options=list(category_levels["Occupation"]),
                index=None,
                placeholder="Select your occupation",
                key=FORM_WIDGET_KEYS["Occupation"],
            )

        if "BMI Category" in categorical_input_features:
            st.selectbox(
                "BMI category",
                options=list(category_levels["BMI Category"]),
                index=None,
                placeholder="Select a category or calculate your BMI above",
                key=FORM_WIDGET_KEYS["BMI Category"],
                format_func=format_bmi_category,
                help=(
                    "BMI is a general measurement based on your height and weight. "
                    "If you are unsure, use the BMI helper above to find your category."
                )
            )

    with sleep_column:
        st.subheader("Sleep and daily habits")

        sleep_features = [
            feature
            for feature in (
                "Sleep Duration",
                "Quality of Sleep",
                "Physical Activity Level",
            )
            if feature in numerical_input_features
        ]

        range_descriptions = {
            "Sleep Duration": (
                "Enter your usual sleep duration from 5.8 to 8.5 hours."
            ),
            "Quality of Sleep": (
                "Choose a rating from 4 to 9, where 4 means lower sleep "
                "quality and 9 means higher sleep quality."
            ),
            "Physical Activity Level": (
                "Enter your daily physical activity from 30 to 90 minutes."
            ),
        }

        for feature in sleep_features:
            render_number_input(feature)

            if feature in range_descriptions:
                st.caption(range_descriptions[feature])

    if health_features:
        st.divider()
        st.subheader("Health readings")
        st.info(
            "Can't measure these readings right now? You may leave the example "
            "values shown: 130 mmHg for your upper blood pressure number and "
            "68 beats per minute for your resting heart rate."
        )

        st.caption(
            "These are common values from the dataset, not your personal health "
            "measurements. Using them may make your sleep pattern result less "
            "accurate."
        )

        health_columns = st.columns(len(health_features))
        for health_column, feature in zip(
            health_columns,
            health_features,
        ):
            with health_column:
                render_number_input(feature)

    submitted = st.form_submit_button(
        "Check sleep pattern",
        type="primary",
        use_container_width=True,
    )


raw_input = {
    feature: st.session_state.get(FORM_WIDGET_KEYS[feature])
    for feature in FORM_WIDGET_KEYS
    if feature in active_features
}

if submitted:
    validation_errors = validate_input(
        raw_input,
        model_package,
    )

    if validation_errors:
        st.session_state.pop("screening_result", None)

        error_list = "\n".join(
            f"- {validation_error}"
            for validation_error in validation_errors
        )

        st.error(
            "Please review the following details before continuing:\n\n"
            f"{error_list}"
        )

    else:
        try:
            with st.spinner("Preparing your sleep pattern result..."):
                prepared_input = prepare_model_input(
                    raw_input,
                    model_package,
                )

                prediction = model.predict(prepared_input)[0]
                probabilities = model.predict_proba(
                    prepared_input
                )[0]

                if len(probabilities) != len(model.classes_):
                    raise ValueError(
                        "Some information needed to prepare the result is missing."
                    )

                if prediction not in model_package["target_classes"]:
                    raise ValueError(
                        "The result could not be matched to an available sleep pattern."
                    )

                probability_table = pd.DataFrame(
                    {
                        "Class": model.classes_,
                        "Probability (%)": probabilities * 100,
                    }
                ).sort_values(
                    by="Probability (%)",
                    ascending=False,
                )

                highest_probability = float(
                    probability_table.iloc[0]["Probability (%)"]
                )

                st.session_state["screening_result"] = {
                    "prediction": prediction,
                    "highest_probability": highest_probability,
                    "probability_table": probability_table,
                    "raw_input": raw_input.copy(),
                }

        except Exception as prediction_error:
            st.session_state.pop("screening_result", None)

            print(
                "Prediction error: "
                f"{type(prediction_error).__name__}: "
                f"{prediction_error}"
            )

            st.error(
                "We couldn't prepare your result right now. "
                "Please check your answers and try again."
            )


screening_result = st.session_state.get("screening_result")
if screening_result:
    prediction = screening_result["prediction"]
    highest_probability = screening_result[
        "highest_probability"
    ]
    probability_table = screening_result["probability_table"]
    submitted_input = screening_result["raw_input"]

    st.success("Your sleep pattern result is ready.")
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Your sleep pattern result</div>
            <p class="result-value">{prediction}</p>
            <div class="small-note">
                Match level: {highest_probability:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_guidance = {
        "No Sleep Disorder": (
            "Your information does not show strong signs commonly linked "
            "to Insomnia or Sleep Apnea."
        ),
        "Insomnia": (
            "Your information shows a sleep pattern that may be linked "
            "to Insomnia."
        ),
        "Sleep Apnea": (
            "Your information shows a sleep pattern that may be linked "
            "to Sleep Apnea."
        ),
    }

    st.info(
        result_guidance.get(
            prediction,
            "Your information most closely matches this sleep pattern. "
            "Please remember that this result is only a prediction and not a medical diagnosis.",
        )
    )

    result_column, summary_column = st.columns(2)

    with result_column:
        st.subheader("Class probabilities")
        st.bar_chart(
            probability_table.set_index("Class")[
                "Probability (%)"
            ]
        )
        st.dataframe(
            probability_table.round(2),
            use_container_width=True,
            hide_index=True,
        )

    with summary_column:
        st.subheader("Submitted information")
        input_summary = pd.DataFrame(
            {
                "Input": [
                    FEATURE_LABELS.get(feature, feature)
                    for feature in submitted_input
                ],
                "Value": list(submitted_input.values()),
            }
        )
        st.dataframe(
            input_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download result summary",
            data=build_text_summary(screening_result),
            file_name="sleep_pattern_result.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("What does this result mean?"):
        st.write(
            "This result shows the sleep pattern that best matches the information "
            "you entered. The percentages show how closely your information matches "
            "each of the three possible sleep patterns."
        )
        st.write(
            "Please remember that this is only a prediction and not a medical "
            "diagnosis. If you are worried about your sleep or health, consider "
            "speaking with a healthcare professional."
        )


st.divider()
st.caption(
    "Sleep Pattern Checker"
)