import metrics.metrics as metrics
from metrics.metrics import AccuracyMeter, AverageValueMeter, ConfusionMeter, F1Meter, SegmentationConfusionMeter

from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import itertools
import io
import logging


class MetricsManager:
    def __init__(self, hparams, k):
        """
        Manages metrics histories for training and validation
        :param hparams: dictionary with all hyper parameters of the training procedure
        :param k: number of classes in the task
        """
        self._k = k
        self._hparams = hparams
        self._train_metrics = []
        self._valid_metrics = []
        self._best_loss = np.inf
        self._best_metrics = None

    def init_with_history(self, train_metrics_history, valid_metrics_history,
                          best_loss, best_metrics):
        """
        Initialize metrics based on history from a previous run
        :param train_metrics_history: training metrics history (list of metrics dictionaries)
        :param valid_metrics_history: validation metrics history (list of metrics dictionaries)
        :param best_loss: best loss from the previous run
        :param best_metrics: metrics snapshot when best loss was taken
        :return: None
        """
        self._train_metrics = train_metrics_history
        self._valid_metrics = valid_metrics_history
        self._best_loss = best_loss
        self._best_metrics = best_metrics

    def add_epoch(self):
        """
        Recreates metrics and adds them into the histories
        :return: None
        """
        self._update_best_metrics()

        cur_train_metrics = self._init_metrics(self._hparams)
        cur_valid_metrics = self._init_metrics(self._hparams)

        self._train_metrics.append(cur_train_metrics)
        self._valid_metrics.append(cur_valid_metrics)

    def update_epoch_train_metrics(self, loss, predicted, target):
        """
        Adds new measurements into current training metrics
        :param loss: loss value calculated on the current batch
        :param predicted: Can be an N x K tensor of predicted scores obtained from
            the model for N examples and K classes or an N-tensor of
            integer values between 0 and K-1.
        :param target: Can be a N-tensor of integer values assumed to be integer
            values between 0 and K-1 or N x K tensor, where targets are
            assumed to be provided as one-hot vectors
        :return: None
        """
        cur_train_metrics = self._train_metrics[-1]

        for metric_info in cur_train_metrics:
            m_name = metric_info['name']
            train_metric = metric_info['metric']

            if m_name == 'loss':
                train_metric.add(loss)
            else:
                train_metric.add(predicted, target)

    def update_epoch_valid_metrics(self, loss, predicted, target):
        """
        Adds new measurements into current validation metrics
        :param loss: loss value calculated on the current batch
        :param predicted: Can be an N x K tensor of predicted scores obtained from
            the model for N examples and K classes or an N-tensor of
            integer values between 0 and K-1.
        :param target: Can be a N-tensor of integer values assumed to be integer
            values between 0 and K-1 or N x K tensor, where targets are
            assumed to be provided as one-hot vectors
        :return: None
        """
        cur_valid_metrics = self._valid_metrics[-1]

        for metric_info in cur_valid_metrics:
            m_name = metric_info['name']
            valid_metric = metric_info['metric']

            if m_name == 'loss':
                valid_metric.add(loss)
            else:
                valid_metric.add(predicted, target)

    def get_metrics_history(self):
        """
        Returns the whole metrics history
        :return: train_metrics, valid_metrics, best_loss, best_metrics
        """
        return self._train_metrics, self._valid_metrics, self._best_loss, self._best_metrics

    def get_cur_metrics(self):
        """
        Returns metrics for the current epoch
        :return: training metrics, validation metrics
        """
        cur_train_metrics, cur_valid_metrics = [], []

        if len(self._train_metrics) > 0:
            cur_train_metrics = self._train_metrics[-1]
        if len(self._valid_metrics) > 0:
            cur_valid_metrics = self._valid_metrics[-1]

        return cur_train_metrics, cur_valid_metrics

    def is_current_best(self):
        """
        Checks whether metrics for the current epoch are best (based on valid loss)
        among all other metrics in history
        :return: True if current metrics are best
        """
        _, cur_valid_loss = self.get_cur_loss()

        return cur_valid_loss < self._best_loss

    def get_cur_loss(self):
        """
        Returns training and validation losses for the current epoch
        :return: training loss, validation loss
        """
        train_loss, valid_loss = np.inf, np.inf

        def _find_loss_metric(metrics_list):
            for metric_info in metrics_list:
                if metric_info['name'] == 'loss':
                    return metric_info['metric']
            return None

        if len(self._train_metrics) > 0:
            cur_train_metrics = self._train_metrics[-1]
            train_loss_metric = _find_loss_metric(cur_train_metrics)
            if train_loss_metric is not None:
                train_loss = train_loss_metric.value()[0]

        if len(self._valid_metrics) > 0:
            cur_valid_metrics = self._valid_metrics[-1]
            valid_loss_metric = _find_loss_metric(cur_valid_metrics)
            if valid_loss_metric is not None:
                valid_loss = valid_loss_metric.value()[0]

        return train_loss, valid_loss

    def _update_best_metrics(self):
        """
        Updates best metrics if the metrics for current epoch are the best
        :return: None
        """
        _, cur_valid_loss = self.get_cur_loss()
        if cur_valid_loss < self._best_loss:
            self._best_loss = cur_valid_loss
            self._best_metrics = self._valid_metrics[-1]

    def _init_metrics(self, hparams):
        """
        Generates new list of metrics including loss metric
        :param hparams: dictionary with all hyper parameters of the training procedure
        :return: list of metrics
        """
        metrics_list = self.parse_metrics_from_hparams(hparams)
        loss_metric = {'name': 'loss', 'metric': AverageValueMeter(override_name='loss')}
        metrics_list.append(loss_metric)

        return metrics_list

    def parse_metrics_from_hparams(self, hparams):
        """
        Parses and instantiates list of metrics for given hparams
        :param hparams: dictionary with all hyper parameters of the training procedure
        :return: list of metrics
        """
        if 'metrics' not in hparams:
            return []

        metrics_list = []

        for metric_conf in hparams['metrics']:
            m_name = metric_conf['name']
            m_params = metric_conf['params']

            if m_params is not None:
                metric = metrics.__dict__[m_name](k=self._k, **m_params)
            else:
                metric = metrics.__dict__[m_name](k=self._k)

            metrics_list.append({'name': m_name, 'metric': metric})

        return metrics_list


class TensorboardLogger:
    def __init__(self, metrics_manager, logdir, target_label_names):
        """
        Adds metrics for tensorboardX visualization
        :param metrics_manager: MetricsManager instance
        :param logdir: log directory for Tensorboad logs
        :param target_label_names: list of classes of the target variable
        """
        self._metrics_manager = metrics_manager
        self._summary_writer = SummaryWriter(logdir)
        self._target_label_names = target_label_names

        self._initialize_from_history()

    def _initialize_from_history(self):
        """
        Initializes Tensorboard metrics from previous observations found in MetricsManager
        :return: None
        """
        logging.info('Initializing Tensorboard from history...')
        train_metrics_history, valid_metrics_history, _, _ = self._metrics_manager.get_metrics_history()
        for epoch, train_metrics in enumerate(train_metrics_history):
            self._add_metrics(train_metrics, epoch, 'train')
        for epoch, valid_metrics in enumerate(valid_metrics_history):
            self._add_metrics(valid_metrics, epoch, 'valid')
        logging.info('Finished Tensorboard initialization')

    def update_metrics(self, epoch):
        """
        Plots metric values for a new epoch
        (be careful, epoch number should increase according to training epochs)
        :param epoch: index of an epoch (int)
        :return: None
        """
        train_metrics, valid_metrics = self._metrics_manager.get_cur_metrics()
#         if (epoch != len(train_metrics) - 1) or (epoch != len(valid_metrics) - 1):
#             logging.warning('Tensorboard. Epoch number not equal to len of metrics list')

        self._add_metrics(train_metrics, epoch, 'train')
        self._add_metrics(valid_metrics, epoch, 'valid')

    def _add_metrics(self, epoch_metrics, epoch, mode):
        """
        Adds metrics to Tensorboard
        :param epoch_metrics: dict with epoch metrics
        :param epoch: number of epoch
        :param mode: train (for training) or val (for validation)
        :return: None
        """
        for metric_info in epoch_metrics:
            metric = metric_info['metric']
            m_name = metric.name()
            m_value = MetricsParser.get_val_from_metric(metric)
            if type(metric) is ConfusionMeter or type(metric) is SegmentationConfusionMeter:
                m_value = self.plot_confusion_matrix(m_value,
                                                list(self._target_label_names.keys()),
                                                title='{} confusion matrix'.format(mode.capitalize()),
                                                normalize=True)
                self._summary_writer.add_image('{}/{}'.format(mode, m_name), m_value, epoch + 1)
            else:
                self._summary_writer.add_scalar('{}/{}'.format(mode, m_name), m_value, epoch + 1)

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.rcParams['figure.figsize'] = max(len(classes), 12) / 2, max(len(classes), 12) / 2

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size=max(len(classes) // 2, 20))
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, size=16)
        plt.yticks(tick_marks, classes, size=16)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Истинное значение', size=max(len(classes) // 3, 20))
        plt.xlabel('Предсказанное значение', size=max(len(classes) // 3, 20))
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close()

        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        return image

    def close(self):
        """
        Closes SummaryWriter
        :return: None
        """
        self._summary_writer.close()


class MetricsParser:
    @staticmethod
    def get_val_from_metric(metric):
        """
        Converts the metric value to a single number.
        :param metric: metric which value needs to be taken
        :return: metric value
        """
        if type(metric) is AverageValueMeter:
            m_value = metric.value()[0]
        else:
            m_value = metric.value()

        return m_value

    @staticmethod
    def read_metrics_values(epoch_metrics):
        """
        Converts metrics to a dictionary of {metrics_name:value}
        :param epoch_metrics: metrics to convert
        :return: dict of metric values
        """
        return {m_name: MetricsParser.get_val_from_metric(metric)
                for m_name, metric in epoch_metrics.items()}
