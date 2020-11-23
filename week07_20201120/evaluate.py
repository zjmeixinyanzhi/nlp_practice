import sys
import io
import json
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append('%s/data') % CUR_DIR
from evaluate import evaluate as src_evaluate
from collections import OrderedDict

def evaluate(filename):
    """
    评测函数（官方提供评测脚本evaluate.py）
    """
    predict_to_file(filename, filename + '.pred.json')
    ref_ans = json.load(io.open(filename))
    pred_ans = json.load(io.open(filename + '.pred.json'))
    F1, EM, TOTAL, SKIP = src_evaluate(ref_ans, pred_ans)
    output_result = OrderedDict()
    output_result['F1'] = '%.3f' % F1
    output_result['EM'] = '%.3f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    return output_result

class Evaluator(keras.callbacks.Callback):
    """
    评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(
            os.path.join(data_dir,'dev.json')
            # os.path.join(data_dir,'demo_dev.json')
        )
        if float(metrics['F1']) >= self.best_val_f1:
            self.best_val_f1 = float(metrics['F1'])
            model.save_weights(os.path.join(output_dir,'roberta_best_model.weights'))
            model.save(os.path.join(output_dir,'roberta_best_model.h5'))
        metrics['BEST_F1'] = self.best_val_f1
        print(metrics)