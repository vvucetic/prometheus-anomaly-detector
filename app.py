"""docstring for packages."""
import time
import os
import logging
from datetime import datetime
from multiprocessing import Process, Queue
from queue import Empty as EmptyQueueException
import tornado.ioloop
import tornado.web
from prometheus_client import Gauge, generate_latest, REGISTRY
from prometheus_api_client import PrometheusConnect, Metric
from configuration import Configuration
import model
import schedule

# Set up logging
_LOGGER = logging.getLogger(__name__)

METRICS_LIST = Configuration.metrics_list

# list of ModelPredictor and Gauge Objects shared between processes
MODEL_LIST = list()

pc = PrometheusConnect(
    url=Configuration.prometheus_url,
    headers=Configuration.prom_connect_headers,
    disable_ssl=True,
)

class MainHandler(tornado.web.RequestHandler):
    """Tornado web request handler."""

    def initialize(self, data_queue, gauge_dict):
        """Check if new predicted values are available in the queue before the get request."""
        try:
            model_list = data_queue.get_nowait()
            self.settings["model_list"] = model_list
            self.settings["gauge_dict"] = gauge_dict

            # add new gauges
            fresh_metrics = set([predictor.metric.metric_name for predictor in model_list])
            for predictor in model_list:
                unique_metric = predictor.metric
                if unique_metric.metric_name not in gauge_dict:
                    label_list = list(unique_metric.label_config.keys())
                    label_list.append("value_type")
                    gauge_dict[unique_metric.metric_name] = Gauge(
                        unique_metric.metric_name + "_" + predictor.model_name,
                        predictor.model_description,
                        label_list,
                    )
            # remove old gauges
            for garbage_gauge in filter(lambda x: x not in fresh_metrics, gauge_dict.keys()):
                del gauge_dict[garbage_gauge]
        except EmptyQueueException:
            pass

    async def get(self):
        """Fetch and publish metric values asynchronously."""
        # update metric value on every request and publish the metric
        gauge_dict = self.settings["gauge_dict"]
        for predictor_model in self.settings["model_list"]:
            # get the current metric value so that it can be compared with the
            # predicted values
            try:
                current_metric_value = Metric(
                    pc.get_current_metric_value(
                        metric_name=predictor_model.metric.metric_name,
                        label_config=predictor_model.metric.label_config,
                    )[0]
                )
            except IndexError:
                # metric no longer available, skip it
                continue

            metric_name = predictor_model.metric.metric_name
            prediction = predictor_model.predict_value(datetime.now())

            # Check for all the columns available in the prediction
            # and publish the values for each of them
            for column_name in list(prediction.columns):
                gauge_dict[metric_name].labels(
                    **predictor_model.metric.label_config, value_type=column_name
                ).set(prediction[column_name][0])

            # Calculate for an anomaly (can be different for different models)
            anomaly = 1
            if (
                current_metric_value.metric_values["y"][0] < prediction["yhat_upper"][0]
            ) and (
                current_metric_value.metric_values["y"][0] > prediction["yhat_lower"][0]
            ):
                anomaly = 0

            # create a new time series that has value_type=anomaly
            # this value is 1 if an anomaly is found 0 if not
            gauge_dict[metric_name].labels(
                **predictor_model.metric.label_config, value_type="anomaly"
            ).set(anomaly)

        self.write(generate_latest(REGISTRY).decode("utf-8"))
        self.set_header("Content-Type", "text; charset=utf-8")


def make_app(data_queue):
    """Initialize the tornado web app."""
    _LOGGER.info("Initializing Tornado Web App")
    gauge_dict = dict()
    return tornado.web.Application(
        [
            (r"/metrics", MainHandler, dict(data_queue=data_queue, gauge_dict=gauge_dict)),
            (r"/", MainHandler, dict(data_queue=data_queue, gauge_dict=gauge_dict)),
        ]
    )


def refresh_unique_metrics():
    predictor_list = list()
    gauge_dict = dict()
    for metric in METRICS_LIST:
        # Initialize a predictor for all metrics first
        metric_init = pc.get_current_metric_value(metric_name=metric)

        for unique_metric in metric_init:
            predictor_list.append(
                model.MetricPredictor(
                    unique_metric,
                    rolling_data_window_size=Configuration.rolling_training_window_size,
                    changepoint_prior_scale=Configuration.changepoint_prior_scale,
                    cap=Configuration.fbp_cap,
                    floor=Configuration.fbp_floor
                )
            )
    return predictor_list


def train_model(initial_run=False, data_queue=None):
    """Train the machine learning model."""

    # Refresh metrics list
    _LOGGER.info("Refreshing metrics list.")
    predictor_list = refresh_unique_metrics()
    
    for predictor_model in predictor_list:
        metric_to_predict = predictor_model.metric
        data_start_time = datetime.now() - Configuration.metric_chunk_size
        if initial_run:
            data_start_time = (
                datetime.now() - Configuration.rolling_training_window_size
            )

        # Download new metric data from prometheus
        new_metric_data = pc.get_metric_range_data(
            metric_name=metric_to_predict.metric_name,
            label_config=metric_to_predict.label_config,
            start_time=data_start_time,
            end_time=datetime.now(),
        )[0]

        # Train the new model
        start_time = datetime.now()
        predictor_model.train(
            new_metric_data, Configuration.retraining_interval_minutes
        )
        _LOGGER.info(
            "Total Training time taken = %s, for metric: %s %s",
            str(datetime.now() - start_time),
            metric_to_predict.metric_name,
            metric_to_predict.label_config,
        )

    data_queue.put(predictor_list)


if __name__ == "__main__":
    # Queue to share data between the tornado server and the model training
    model_queue = Queue()

    # Initial run to generate metrics, before they are exposed
    train_model(initial_run=True, data_queue=model_queue)

    # Set up the tornado web app
    app = make_app(model_queue)
    app.listen(8080)
    server_process = Process(target=tornado.ioloop.IOLoop.instance().start)
    # Start up the server to expose the metrics.
    server_process.start()

    # Schedule the model training
    schedule.every(Configuration.retraining_interval_minutes).minutes.do(
        train_model, initial_run=False, data_queue=model_queue
    )
    _LOGGER.info(
        "Will retrain model every %s minutes", Configuration.retraining_interval_minutes
    )

    while True:
        schedule.run_pending()
        time.sleep(1)

    # join the server process in case the main process ends
    server_process.join()
