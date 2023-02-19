import pandas as pd
import torch
from tqdm import tqdm
from config import Config
from train_fine_tune import list2ts2device, softmax
from transformers import BertTokenizer
from utils import DataIterator
import numpy as np
import logging
import os

gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_test(test_iter, model_file):
    model = torch.load(model_file)  # 读取保存的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("***** Running Prediction *****")
    logger.info("  Predict Path = %s", model_file)
    model.eval()
    pred_label_list = []
    for input_ids, input_mask, segment_ids, cls_label, seq_length in tqdm(
            test_iter):
        input_ids = list2ts2device(input_ids)
        input_mask = list2ts2device(input_mask)
        segment_ids = list2ts2device(segment_ids)
        y_preds,_ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        cls_pred = y_preds.detach().cpu()
        cls_probs = softmax(cls_pred.numpy())
        cls_pre = np.argmax(cls_probs, axis=-1)
        pred_label_list += list(cls_pre)
    print(len(pred_label_list))
    # print(pred_label_list)
    # print(true_label_list)

    with open('result.txt', 'w') as fw:
        for pred in pred_label_list:
            fw.write(Config().label_name[pred])


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    do_lower_case = False
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size,
                            'test.csv',
                            use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    set_test(dev_iter, config.checkpoint_path)
