from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Cars MML-SG projects"
    API_V1_STR: str = "/api/v1"

    MODEL_PATH: str = "ml_model"

    MODELS_NAMES: tuple = ("anomalies_nearest_neighbors.pkl",
                           "preprocess_ad_imputer_model.pkl",
                           "preprocess_ad_normalization_model.pkl",
                           "preprocess_ad_ourlier_detection_model.pkl",
                           "preprocess_regression_imputer_model.pkl",
                           "preprocess_regression_normalization_model.pkl",
                           "preprocess_regression_ourlier_detection_model.pkl",
                           "properties_anomaly_detection_model.keras",
                           "regression_price_model.pkl")

    class Config:
        case_sensitive = True


settings = Settings()
