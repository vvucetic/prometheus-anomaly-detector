import datetime
import pandas
from fbprophet import Prophet
from metric import Metric
from prometheus_client import Gauge


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "prophet"
    model_description = "Forecasted value from Prophet model"
    model = None
    predicted_df = None
    metric = None
    predicted_metric_gauge = None

    def __init__(self, metric):
        self.metric = Metric(metric)

    def train(self, metric_data, prediction_duration=15, prediction_freq="1MIN"):
        # convert incoming metric to Metric Object
        self.metric = self.metric + Metric(metric_data)  # !!!!!! Memory bloat !!!!

        # Don't really need to store the model, as prophet models are not retrainable
        # But storing it as an example for other models that can be retrained
        self.model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)

        self.model.fit(self.metric.metric_values)
        future = self.model.make_future_dataframe(
            periods=int(prediction_duration), freq=prediction_freq
        )
        forecast = self.model.predict(future)
        forecast["timestamp"] = forecast["ds"]
        forecast = forecast[["timestamp", "yhat", "yhat_lower", "yhat_upper"]]
        forecast = forecast.set_index("timestamp")
        self.predicted_df = forecast
        print(forecast)

    def predict_value(self, prediction_datetime):
        """
        This function returns the predicted value of the metric for the prediction_datetime
        """
        nearest_index = self.predicted_df.index.get_loc(prediction_datetime, method="nearest")
        return self.predicted_df.iloc[[nearest_index]]


def get_df_from_metric_json(metric):
    """
    Method to convert a json object of a Prometheus metric to a dictionary of shaped Pandas DataFrames

    The shape is dict[metric_metadata] = Pandas Object

    Pandas Object = timestamp, value
                    15737933, 1
                    .....

    This method can also be used to update an existing dictionary with new data
    """
    df = pandas.DataFrame(metric["values"], columns=["ds", "y"]).apply(
        pandas.to_numeric, args=({"errors": "coerce"})
    )
    df["ds"] = pandas.to_datetime(df["ds"], unit="s")

    return df