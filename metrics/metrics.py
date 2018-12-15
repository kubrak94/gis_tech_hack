import numpy as np
import torch


class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """
    def __init__(self, override_name='default_meter_name'):
        self._override_name = override_name

    def reset(self):
        """Resets the meter to default settings."""
        pass

    def add(self, *value):
        """Log a new value to the meter

        :param value: Updates to include
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass

    def name(self):
        return self._override_name

    
class AverageValueMeter(Meter):
    def __init__(self, override_name='average_value'):
        """
        Maintains value updates and counts mean and std of these updates
        """
        super(AverageValueMeter, self).__init__(override_name)
        self.reset()
        self.val = 0

    def add(self, value):
        """
        Updates mean and std for value history given a new value
        :param value: value to add for averaging
        :return: None
        """
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += 1

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - 1 * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        """
        Returns mean and std for the value history
        :return: mean, std
        """
        return self.mean, self.std

    def reset(self):
        """
        Resets the metric to initial state
        :return:
        """
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
    

class ConfusionMeter(Meter):
    def __init__(self, k, normalized=False, override_name='confusion_matrix'):
        """Maintains a confusion matrix for a given calssification problem.

        The ConfusionMeter constructs a confusion matrix for a multi-class
        classification problems. It does not support multi-label, multi-class problems:
        for such problems, please use MultiLabelConfusionMeter.

        :param k: number of classes in the classification problem
        :param normalized: Determines whether or not the confusion matrix
            is normalized or not

        """
        super(ConfusionMeter, self).__init__(override_name=override_name)
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        """
        Resets internal state of the metric to initial state
        :return: None
        """
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        :param predicted: Can be an N x K tensor of predicted scores obtained from
            the model for N examples and K classes or an N-tensor of
            integer values between 0 and K-1.
        :param target: Can be a N-tensor of integer values assumed to be integer
            values between 0 and K-1 or N x K tensor, where targets are
            assumed to be provided as one-hot vectors
        """
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        :return: Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


# ============= Segmentation metrics =============


class SegmentationConfusionMeter(ConfusionMeter):
    def __init__(self, k, normalized=False, override_name='segmentation_confusion_meter'):
        """Constructs a confusion matrix for a multi-class semantic segmentation problem.

        :param k: number of classes in the problem
        :param normalized: Determines whether the confusion matrix is normalized
        """
        super(SegmentationConfusionMeter, self).__init__(k, normalized, override_name=override_name)
        self.conf = np.ndarray((k, k), dtype=np.int64)
        self.reset()

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is number of classes

        :param: predicted: Can be an N x K x H x W tensor of predicted scores obtained from
            the model for N examples and K classes
        :param target: Can be an N x H x W tensor of target scores for N examples
            and K classes where each value is 0 <= target[i] <= K - 1
        """
        predicted = predicted.argmax(1)
        predicted = predicted.view(torch.numel(predicted))
        target = target.view(torch.numel(target))
        super(SegmentationConfusionMeter, self).add(predicted, target)


class MeanIntersectionOverIUnionMeter:
    def __init__(self, k, weighted=False, override_name='mean_intersection_over_union'):
        """Calculates mean intersection over union for a multi-class semantic segmentation problem.
        The meter makes calculations based on confusion matrix

        :param k: number of classes in the problem
        """
        assert k >= 2, 'Number of classes must be >= 2'

        self._segm_conf_meter = SegmentationConfusionMeter(k, normalized=False, override_name=override_name)
        self._weighted = weighted
        self._miou = 0.
        self._override_name = override_name
        self.reset()

    def add(self, predicted, target):
        """Computes mean intersection over union K classes semantic segmentation problem.

        :param predicted: Can be an N x K x H x W tensor of predicted scores obtained from
            the model for N examples and K classes
        :param target: Can be an N x H x W tensor of target scores for N examples
            and K classes where each value is 0 <= target[i] <= K - 1
        :return: None
        """
        self._segm_conf_meter.add(predicted, target)
        self._miou = self.calc_miou(self._segm_conf_meter.value())

    def reset(self):
        """
        Resets the meter to initial state
        :return: None
        """
        self._segm_conf_meter.reset()
        self._miou = 0.

    def calc_miou(self, conf_matrix):
        """
        Computes mean intersection over union for a given confusion matrix
        :param conf_matrix: K x K non-normalized confusion matrix where K is a number of classes
        :return: mean intersection over union float value
        """
        tp = np.diagonal(conf_matrix)
        pos_pred = conf_matrix.sum(axis=0)
        pos_gt = conf_matrix.sum(axis=1)

        # Check which classes have elements
        valid_idxs = pos_gt > 0
        ious_valid = np.logical_and(valid_idxs, pos_gt + pos_pred - tp > 0)

        # Calculate intersections over union for each class
        ious = np.full((conf_matrix.shape[0]), np.NaN)
        ious[ious_valid] = np.divide(tp[ious_valid],
                                     pos_gt[ious_valid] + pos_pred[ious_valid] - tp[ious_valid])

        # Calculate mean intersection over union
        if not self._weighted:
            miou = np.mean(ious[ious_valid])
        else:
            weights = np.divide(pos_gt, conf_matrix.sum())
            miou = np.sum(ious[ious_valid] * weights[ious_valid])

        return miou

    def value(self):
        """
        Returns current value of the mean intersection over union
        :return: miou (float)
        """
        return self._miou

    def name(self):
        return self._override_name
