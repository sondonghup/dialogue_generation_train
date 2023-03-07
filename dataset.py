import os
import json
import random
import re
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def load_dataset(menu, data_dir, sep_token: str, eos_token: str):
    if menu == 'ai':
        all_utterances = list()

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        for file in os.listdir(data_dir):
            if file.endswith('json'):
                with open(f'{data_dir}{file}', encoding='utf-8')as f:
                    file_datas = json.load(f)
                print(f'======= {file} =======')
                for file_data in tqdm(file_datas['data']):
                    participant_id = None
                    utterances = str()

                    if file_data['header']['dialogueInfo']['numberOfParticipants'] != 2:
                        continue
                    
                    for utterance in file_data['body']:
                        if participant_id is None:
                            utterances += utterance['utterance']
                        elif participant_id == utterance['participantID']:
                            utterances += ' ' + utterance['utterance']
                        else:
                            utterances += sep_token + utterance['utterance']
                        
                        participant_id = utterance['participantID']
                    utterances += eos_token
                    all_utterances.append(utterances)

        return all_utterances
    

    if menu == 'kakao':
        dialogue_list = []
        all_utterances = []
        for file in os.listdir(data_dir):
            if file.endswith('txt'):
                print(f'======= {file} =======')
                with open(f'{data_dir}{file}', encoding='utf-8')as f:
                  for line in tqdm(f):
                    try:
                      dialogue = line.split(',')[1].split(':')[0].strip()
                      if dialogue != '사진' and dialogue != '이모티콘' and dialogue.find('http') == -1:
                        dialogue_list.append(dialogue)
                    except:
                      continue
                  for i in range(1, int(len(dialogue_list) / 10), 1):
                      utterances = sep_token.join(dialogue_list[4+(i - 1)*10 : i*10]) + eos_token
                      all_utterances.append(utterances)

        return all_utterances

class DialogueDataset(Dataset):
    def __init__(self, datas, tokenizer) -> None:
        super(DialogueDataset, self).__init__()
        self.tokenizer = tokenizer
        self.input_ids = list()
        self.attention_mask = list()
        self.labels = list()

        print(f"model_max_length : {self.tokenizer.model_max_length}")

        for data in tqdm(datas):
            encode_data = self.tokenizer(data, truncation = True, max_length = tokenizer.model_max_length)
            # print(f'encode_data : {encode_data}')

            self.input_ids.append(encode_data['input_ids'])
            self.attention_mask.append(encode_data['attention_mask'])
            self.labels.append(encode_data['input_ids'])
            # print(f'data : {data}')
            # print(f'input_ids : {self.input_ids[-1]}')
            # print(f'attention_mask : {self.attention_mask[-1]}')
            # print(f'labels : {self.labels[-1]}')
            # input()


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

def collate_fn(batch, pad_token_id, bos_token_id):
    def seq_length_(p):
        return len(p[0])

    # print(f'batch : {type(batch)}')  
    # print(f'pad_token_id : {pad_token_id}')         
    # print(f'seq_length : {seq_length_}')

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_seq_size = len(max_seq_sample)

    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    attention_masks = torch.zeros(batch_size, max_seq_size).fill_(bos_token_id).long()
    labels = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    # input_ids = torch.full((batch_size, max_seq_size), pad_token_id).long()
    # attention_masks = torch.full((batch_size, max_seq_size), pad_token_id).long()
    # labels = torch.full((batch_size, max_seq_size), pad_token_id).long()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_attention_masks = sample[1]
        sample_labels = sample[2]

        # print(f'sample_input_ids : {sample_input_ids}')  
        # print(f'sample_attention_masks : {sample_attention_masks}')         
        # print(f'sample_labels : {sample_labels}')

        # input()

        '''
        y = tensor.new_tensor(x, requires_grad = True)
        -> 파라미터가 무엇이든 간에 이를 읽어서 leaf variable로 생성한다.
        y = x.clone.detach()
        -> computational graph에서 더이상 필요하지 않을 때 사용할 수 있다. computational graph에서 분리 할때
        -> y를 계산해도 x에 영향 없음 weight를 통해 특정 작업을 하고 싶을때 이용
        y = torch.empty_like(x).copy_(x)
        -> y에 gradient가 흐를수 있음
        y = torch.tensor(x)
        -> 명확하고 빠른 방법. 
        '''

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        attention_masks[idx].narrow(0, 0, len(sample_attention_masks)).copy_(torch.LongTensor(sample_attention_masks))
        labels[idx].narrow(0, 0, len(sample_labels)).copy_(torch.LongTensor(sample_labels))

    return input_ids, attention_masks, labels
