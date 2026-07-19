"""Detection metrics.

``FixedThresholdMetrics`` is the paper-facing implementation used by the
train/test entrypoint.  ``PD_FA`` is retained only for old callers that expect
an eleven-point sweep.
"""

import numpy as np
import torch

from evaluation.component_matching import match_components


class FixedThresholdMetrics:
    """Stream exact pixel/object metrics at one frozen probability threshold."""

    def __init__(
        self,
        threshold=0.5,
        *,
        matching_rule="overlap",
        centroid_distance=3.0,
        connectivity=2,
        min_component_area=1,
    ):
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if matching_rule not in {"overlap", "centroid"}:
            raise ValueError("matching_rule must be overlap or centroid")
        self.threshold = threshold
        self.matching_rule = matching_rule
        self.centroid_distance = float(centroid_distance)
        self.connectivity = int(connectivity)
        self.min_component_area = int(min_component_area)
        self.reset()

    def reset(self):
        self.intersection = 0
        self.union = 0
        self.tp_objects = 0
        self.gt_objects = 0
        self.fp_components = 0
        self.fp_pixels = 0
        self.total_pixels = 0
        self.num_images = 0

    def update(self, probabilities, labels):
        if not isinstance(probabilities, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("probabilities and labels must be torch tensors")
        if probabilities.shape != labels.shape or probabilities.ndim != 4:
            raise ValueError(
                "probabilities and labels must share shape [N, 1, H, W]"
            )
        if probabilities.shape[1] != 1:
            raise ValueError("only one-channel binary predictions are supported")
        scores = probabilities.detach().to(device="cpu").numpy()
        targets = labels.detach().to(device="cpu").numpy()
        if not np.isfinite(scores).all() or not np.isfinite(targets).all():
            raise ValueError("metric inputs contain NaN or infinity")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("probabilities must lie in [0, 1]")
        for score, target in zip(scores[:, 0], targets[:, 0]):
            prediction = score >= self.threshold
            ground_truth = target > 0
            result = match_components(
                prediction,
                ground_truth,
                rule=self.matching_rule,
                centroid_distance=self.centroid_distance,
                connectivity=self.connectivity,
                min_component_area=self.min_component_area,
            )
            self.intersection += int(np.count_nonzero(prediction & ground_truth))
            self.union += int(np.count_nonzero(prediction | ground_truth))
            self.tp_objects += int(result.num_tp_objects)
            self.gt_objects += int(result.num_gt)
            self.fp_components += int(result.num_fp_components)
            self.fp_pixels += int(result.num_fp_pixels)
            self.total_pixels += int(ground_truth.size)
            self.num_images += 1

    def get(self):
        if self.num_images == 0 or self.total_pixels == 0:
            raise ValueError("no images have been evaluated")
        return {
            "mIoU": float(self.intersection / self.union) if self.union else 0.0,
            "Pd": float(self.tp_objects / self.gt_objects) if self.gt_objects else 0.0,
            "Fa_per_million_pixels": float(
                self.fp_pixels / self.total_pixels * 1_000_000.0
            ),
            "Fa_components_per_megapixel": float(
                self.fp_components / (self.total_pixels / 1_000_000.0)
            ),
            "tp_objects": int(self.tp_objects),
            "gt_objects": int(self.gt_objects),
            "fp_components": int(self.fp_components),
            "fp_pixels": int(self.fp_pixels),
            "total_pixels": int(self.total_pixels),
            "num_images": int(self.num_images),
        }

class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([self.bins + 1])
        self.pos_arr  = np.zeros([self.bins + 1])
        self.fp_arr   = np.zeros([self.bins + 1])
        self.neg_arr  = np.zeros([self.bins + 1])
        self.class_pos= np.zeros([self.bins + 1])



class PD_FA():
    """Deprecated sweep wrapper backed by the exact formal matcher.

    New code should use :class:`FixedThresholdMetrics` or
    ``evaluation.threshold_sweep`` so its protocol is explicit in artifacts.
    """

    def __init__(self, nclass, bins, size):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.size = size
        self.reset()

    def update(self, preds, labels):
        probabilities = torch.sigmoid(preds)
        for evaluator in self.evaluators:
            evaluator.update(probabilities, labels)

    def get(self, img_num):
        del img_num
        rows = [evaluator.get() for evaluator in self.evaluators]
        final_fa = np.asarray(
            [row['Fa_per_million_pixels'] / 1_000_000.0 for row in rows]
        )
        final_pd = np.asarray([row['Pd'] for row in rows])
        return final_fa, final_pd

    def reset(self):
        self.evaluators = [
            FixedThresholdMetrics(index / self.bins, matching_rule='overlap')
            for index in range(self.bins + 1)
        ]

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
