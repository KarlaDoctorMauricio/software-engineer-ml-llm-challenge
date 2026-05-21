import pandas as pd

from typing import Tuple, Union, List

TOP_FEATURES = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        df = data.copy()

        # Generate min_diff feature
        fecha_o = pd.to_datetime(
            df["Fecha-O"],
            format="%Y-%m-%d %H:%M:%S"
        )

        fecha_i = pd.to_datetime(
            df["Fecha-I"],
            format="%Y-%m-%d %H:%M:%S"
        )

        df["min_diff"] = (
            fecha_o - fecha_i
        ).dt.total_seconds() / 60

        # Generate target
        df["delay"] = (df["min_diff"] > 15).astype(int)

        # One-hot encoding
        features = pd.concat(
            [
                pd.get_dummies(df["OPERA"], prefix="OPERA"),
                pd.get_dummies(df["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(df["MES"], prefix="MES"),
            ],
            axis=1,
        )

        # Ensure all required columns exist
        for column in TOP_FEATURES:
            if column not in features.columns:
                features[column] = 0

        # Keep only selected features
        features = features[TOP_FEATURES].astype(int)

        # Return target if requested
        if target_column is not None:
            target = df[[target_column]]
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        return