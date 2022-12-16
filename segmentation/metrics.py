from enum import Enum
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import fbeta_score, f1_score, iou_score, accuracy, precision, recall, \
    sensitivity, balanced_accuracy, specificity, positive_predictive_value, negative_predictive_value, \
    false_negative_rate, false_positive_rate, false_omission_rate, false_discovery_rate, positive_likelihood_ratio, \
    negative_likelihood_ratio


class MetricReduction(Enum):
    micro = "micro"
    macro = "macro"
    weighted = "weighted"
    micro_imagewise = "micro-imagewise"
    macro_imagewise = "macro-imagewise"
    weighted_imagewise = "weighted-imagewise"
    none = "none"


# https://en.wikipedia.org/wiki/Confusion_matrix
class Metrics(Enum):
    fbeta_score = 'fbeta_score'
    f1_score = 'f1_score'
    iou_score = 'iou_score'
    accuracy = 'accuracy'
    precision = 'precision'
    recall = 'recall'
    sensitivity = 'sensitivity'
    specificity = 'specificity'
    balanced_accuracy = 'balanced_accuracy'
    positive_predictive_value = 'positive_predictive_value'
    negative_predictive_value = 'negative_predictive_value'
    false_negative_rate = 'false_negative_rate'
    false_positive_rate = 'false_positive_rate'
    false_omission_rate = 'false_omission_rate'
    false_discovery_rate = 'false_discovery_rate'
    positive_likelihood_ratio = 'positive_likelihood_ratio'
    negative_likelihood_ratio = 'negative_likelihood_ratio'

    def get_metric(self):
        return {
            'fbeta_score': fbeta_score,
            'f1_score': f1_score,
            'iou_score': iou_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'positive_predictive_value': positive_predictive_value,
            'negative_predictive_value': negative_predictive_value,
            'false_negative_rate': false_negative_rate,
            'false_positive_rate': false_positive_rate,
            'false_omission_rate': false_omission_rate,
            'false_discovery_rate': false_discovery_rate,
            'positive_likelihood_ratio': positive_likelihood_ratio,
            'negative_likelihood_ratio': negative_likelihood_ratio,
        }[self.value]


if __name__ == "__main__":
    pass
    import segmentation_models_pytorch as smp
