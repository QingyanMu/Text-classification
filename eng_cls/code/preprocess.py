import pandas as pd
from config import Config

config = Config()


# def get_data_csv(file_path, file_name):
#     text_list = []
#     label_list = []
#     with open(file_path + file_name, encoding='utf-8') as fr:
#         for i, line in enumerate(fr):
#             if i > 0:
#                 text, label = line.strip().split('\t')
#                 text_list.append(text)
#                 label_list.append(label)
#     pd_dict = {'text': text_list, 'label': label_list}
#     pd_data = pd.DataFrame(pd_dict)
#     pd_data.to_csv(file_path + 'dev.csv', encoding='utf-8')


# get_data_csv(config.base_dir, 'dev.tsv')
# def get_data_csv_test(file_path, file_name):
#     text_list = []
#     label_list = []
#     with open(file_path + file_name, encoding='utf-8') as fr:
#         for i, line in enumerate(fr):
#             if i > 0:
#                 index, text = line.strip().split('\t')
#                 text_list.append(text)
#                 label_list.append(-1)
#     pd_dict = {'text': text_list, 'label': label_list}
#     pd_data = pd.DataFrame(pd_dict)
#     pd_data.to_csv(file_path + 'dev.csv', encoding='utf-8')


train_df = pd.read_csv('Train.csv', encoding='utf8')
# dev_df = pd.read_csv(config.base_dir + 'dev.csv', encoding='utf8')

print(train_df.head(5))


def cal_text_len(row):
    row_len = len(row.strip().split(' '))
    if row_len < 64:
        return 64
    elif row_len < 128:
        return 128
    elif row_len < 256:
        return 256
    elif row_len < 384:
        return 384
    elif row_len < 512:
        return 512
    else:
        return 1024


def get_label_num(row):
    if row == 'negative':
        return 0
    elif row == 'neutral':
        return 1
    else:
        return 2


train_df['text_len'] = train_df['text'].apply(cal_text_len)
train_df['label'] = train_df['labels'].apply(get_label_num)

new_train_df = train_df[: int(len(train_df) * 0.8)]
new_dev_df = train_df[int(len(train_df) * 0.8):]

new_dev_df.to_csv('dev.csv', encoding='utf8', index=False)
new_train_df.to_csv('train.csv', encoding='utf8', index=False)


test_df = pd.read_csv('Test.csv', encoding='utf8')
test_df['label'] = 0
test_df.to_csv('test.csv', encoding='utf8', index=False)



