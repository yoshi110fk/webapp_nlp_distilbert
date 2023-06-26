import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import DistilBertTokenizerFast, DistilBertModel

# distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# pad_token_idx = tokenizer.pad_token_id

class DistilbertClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # d_model = self.albert.pooler.in_features

        # DISTILBERTの隠れ層の次元数は768, claim1のカテゴリ数が6
        self.fc = nn.Linear(768, 6)

        # # まずは全部OFF
        # for param in self.parameters():
        #     param.requires_grad = False

        # # DistilBERTの最後の層だけ更新ON
        # for param in self.distilbert.transformer.layer[-1].parameters():
        #     param.requires_grad = True

        # # クラス分類のところもON
        # for param in self.fc.parameters():
        #     param.requires_grad = True

    def forward(self, ids, mask):
        out = self.distilbert(ids, mask)
        # [CLS] に対する分散表現のみ取得
        h = out['last_hidden_state']
        h = h[:,0,:]
        h = self.fc(h)
        return h
    
# net = DistilbertClassifier()
# net.load_state_dict(torch.load('./src/lstm.pt'))
# net.load_state_dict(torch.load('./src/distilbert512.pt'))

# net.eval()

# text_train ="A method comprising: delivering a stimulation signal from an implanted electrical signal generator to a structure within a brain via a plurality of electrodes implanted within the structure; and configuring each of the electrodes to have the same polarity to deliver the stimulation signal substantially parallel to axons within the structure, wherein the plurality of electrodes are located on a distal portion of a lead, the method further comprising implanting the distal portion within the structure substantially perpendicular with the axons."
# text_val = "A method of selecting a threshold testing methodology in a cardiac pacing device having a lead for delivering pacing pulses, the method comprising: conducting a threshold test utilizing an algorithmic threshold testing methodology and obtaining a first threshold value; conducting a threshold test utilizing an evoked response methodology and obtaining a second threshold value; comparing the first threshold value with the second threshold value; determining that the lead has a high polarization if the second threshold value is less than the first threshold value and precluding the use of the evoked response methodology so long as the lead has high polarization; and determining that the lead has a low polarization if the first threshold value is substantially equal to or less than the second threshold value and permitting the use of the evoked response methodology."

def tokenizer_ditilbert(text):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    encoded = tokenizer.encode_plus(text, max_length=512, padding='max_length', return_attention_mask=True, truncation=True)
    input_ids = encoded["input_ids"]
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    attention_mask = encoded["attention_mask"]
    attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
    return input_ids, attention_mask
# ids, mask = tokenizer_ditilbert(text_val)

# with torch.no_grad():
#     y = net(ids.unsqueeze(0), mask.unsqueeze(0))
#     y = torch.argmax(F.softmax(y, dim=-1)).detach().numpy()
    
# print('end')