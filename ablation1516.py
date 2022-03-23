from torch_geometric.data import HeteroData
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.nn import ReLU
import torch.nn.functional as F
import torch_geometric.transforms as T, ToUndirected, to_undirected
from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero, GATConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from ast import literal_eval
from collections import Counter
from xgboost import XGBClassifier 
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
import random

import transformers
from transformers import AutoTokenizer, AutoModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int)
parser.add_argument('--ablation', type=int, default = 0)
parser.add_argument('--ratio_words', type=float, default = 0)
parser.add_argument('--chisq_cut', type=float, default = 0)

args = parser.parse_args()
dataset = args.dataset
ablation = args.ablation

print("dataset = ", dataset, "ablation = ", ablation)

ratio_words = args.ratio_words
chisq_cut = args.chisq_cut

# hyperparameter
if dataset == 15:
    if ratio_words == 0:
        ratio_words = 0.2
    if chisq_cut == 0:
        chisq_cut = 99.5
    lrs = 0.003
    h1 = 128
    h2 = 64
    hc = 32
    oc = 16
    
    
if dataset == 16:
    if ratio_words == 0:
        ratio_words = 0.2
    if chisq_cut == 0:
        chisq_cut = 99.75
    lrs = 0.005
    h1 = 128
    h2 = 64
    hc = 32
    oc = 32

# load 
df_tfidf = pd.read_csv("tfidf%s.csv"%(dataset))
document_df = pd.read_csv("document%s.csv"%(dataset))
word_df = pd.read_csv("word%s.csv"%(dataset))
del word_df["url"]
del df_tfidf["url"]
r_user_attribute = pd.read_csv("user_att_scale%s.csv"%(dataset))
interaction_list = pd.read_csv("all_interaction%s.csv"%(dataset), lineterminator='\n')

document_df["clean"] = document_df["clean"].apply(lambda x:literal_eval(x))
document_df["fake"] = document_df["fake"].apply(lambda x: (x == False))

# device = "cpu"
device = "cuda:0"

### keywords filiter ##
fake_target = document_df["fake"].apply(lambda x: int(x*1))

important_model = RandomForestClassifier().fit(df_tfidf, fake_target)

RF_important = pd.DataFrame([df_tfidf.columns,important_model.feature_importances_],index = ["feature","value"]).T.sort_values("value",ascending = False)

all_words = []
for doc in document_df.clean:
    all_words.extend(doc)

distinct_word = np.unique(all_words)

num_words = int(ratio_words * len(distinct_word))
print("ratio_words = ", ratio_words,"num_words = ", num_words,"chisq_cut = ",chisq_cut)

num_most_word = num_words
select_words = RF_important["feature"].iloc[:num_most_word].values.tolist()


df_tfidf = df_tfidf.reset_index(drop=True)
word_df = word_df.loc[:,select_words].reset_index(drop=True)
document_df = document_df.reset_index(drop=True)


## retweeter information ##
interaction_list = interaction_list.loc[interaction_list.iid.isin(document_df.iid),:].reset_index(drop=True)
uniq_interaction_list = interaction_list.drop_duplicates()
interaction_list_group = interaction_list.groupby('uid')
uniq_interaction_list_group = uniq_interaction_list.groupby('uid')


responser_active = dict()
reply_num_list = []

for idx in interaction_list.uid.unique():
    temp = interaction_list_group.get_group(idx)['iid']
    count = Counter(temp)
    num = dict()
    reply_num = 0
    for doc in temp.unique():
        num[doc] = count[doc]
        reply_num += 1
    reply_num_list.append({'responser_id':idx, 'reply_num': reply_num})
    responser_active[idx] = num


df_reply_num = pd.DataFrame(reply_num_list)
df_reply_num = df_reply_num.loc[df_reply_num.reply_num>1].reset_index(drop=True)

distinct_res_id = df_reply_num.responser_id.unique()


# node id
doc_id_to_node = dict()
doc_node_to_id = dict()
word_id_to_node = dict()
word_node_to_id = dict()
select_word_id_to_node = dict()
select_word_node_to_id = dict()
res_id_to_node = dict()
res_node_to_id = dict()


for i,doc in enumerate(document_df.iid):
    doc_id_to_node[doc] = i
    doc_node_to_id[i] = doc
    
for j,word in enumerate(select_words):
    select_word_id_to_node[word] = j
    select_word_node_to_id[j] = word
    
for j,word in enumerate(distinct_word):
    word_id_to_node[word] = j
    word_node_to_id[j] = word
    
for j,res_id in enumerate(distinct_res_id):
    res_id_to_node[res_id] = j
    res_node_to_id[j] = res_id
    

ids = []
for doc in document_df.clean:
    temp = []
    for word in doc:
        temp.append(word_id_to_node[word])
    temp = torch.LongTensor(temp)
    ids.append(temp)

ids = pad_sequence(ids, batch_first = True)

length = torch.LongTensor(document_df['# of words'].values)

num_doc = word_df.shape[0]
num_word = word_df.shape[1]
num_resp = len(distinct_res_id)

# chi-square statistics
df_chisq = pd.read_csv("chisq%s.csv"%(dataset)).sort_values("std_chisq_value")
cut_chisq = np.percentile(df_chisq.std_chisq_value.values, chisq_cut)
df_choose_chisq = df_chisq.loc[df_chisq.std_chisq_value.values > cut_chisq,:]

# connect
connect = []
dw_connect, ww_connect = [], []

for i in word_df.index.tolist():
    for word in word_df.T.loc[word_df.T[i]>0,:].index.tolist():
        dw_connect.append([i,select_word_id_to_node[word]])

for i in range(len(df_choose_chisq)):
    temp = df_choose_chisq.iloc[i]
    try:
        ww_connect.append([select_word_id_to_node[temp.word1],select_word_id_to_node[temp.word2]])
    except:
        continue

doc_res_connect = []
for i in range(len(distinct_res_id)):
    temp = uniq_interaction_list_group.get_group(distinct_res_id[i])
    for j in temp['iid']:
        doc_res_connect.append([res_id_to_node[distinct_res_id[i]], doc_id_to_node[j]])
        


x = np.identity(num_doc)


# pretrained word embedding
from torchtext.vocab import GloVe
embedding_glove = GloVe(name='6B', dim=100)

pretrained_embedding = dict()
for word in word_id_to_node.keys():
    pretrained_embedding[word] = embedding_glove[word].tolist()

word_attr = []
for i in range(num_word):
    word_attr.append(pretrained_embedding[word_node_to_id[i]])
    
responser_attr = np.identity(num_resp)


# train - valid - test
train_lst, test_lst = train_test_split(word_df,test_size = 0.2,shuffle = True,random_state = 29)

train_lst_index = train_lst.index.tolist()
train_ratio = 0.6
train_id = train_lst_index[:int(len(word_df)*train_ratio)]
val_id = train_lst_index[len(train_id):]
test_id = test_lst.index.tolist()

print("train = %s, valid = %s, test = %s"%(len(train_id),len(val_id),len(test_id)))


# bert tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_inputs = tokenizer(document_df.content.values.tolist(), return_tensors='pt', padding=True,truncation=True, max_length=128).to(device)
bert_vocab = np.array(bert_inputs['input_ids'].tolist())
len_bert_vocab = tokenizer.vocab_size
bert_vocab_attr = np.identity(len_bert_vocab)

doc_bert_connect = []
for i in range(bert_vocab.shape[0]):
    for j in bert_vocab[i]:
        if j == 0:
            continue
        doc_bert_connect.append([i,j])

        
# heterogeneous graph        
data = HeteroData()

data['document'].x = torch.FloatTensor(x)
data['document'].y = torch.LongTensor(fake_target)
data['word'].x = torch.FloatTensor(word_attr)
data['responser'].x = torch.FloatTensor(responser_attr)

data['document', 'author', 'word'].edge_index = torch.LongTensor(dw_connect).T
data['word1', 'words', 'word2'].edge_index = torch.LongTensor(ww_connect).T
data['responser', 'reply', 'document'].edge_index = torch.LongTensor(doc_res_connect).T

data = ToUndirected()(data).to(device)
ids = ids.to(device)
bert_ids = bert_inputs['input_ids']


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim1, hidden_dim2, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim1, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim1, hidden_dim2//2, batch_first=True, bidirectional = True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_output, (hidden, cell) = self.lstm(embedded)
        return packed_output

class FusionAttention(nn.Module):
    def __init__(self,dim):
        super(FusionAttention, self).__init__()
        self.attention_matrix = nn.Linear(dim, dim)
        self.project_weight = nn.Linear(dim,1)
    def forward(self, inputs):
        query_project = self.attention_matrix(inputs) # (b,t,d) -> (b,t,d2)
        query_project = F.leaky_relu(query_project)
        project_value = self.project_weight(query_project) # (b,t,h) -> (b,t,1)
        attention_weight = torch.softmax(project_value, dim=1) # Normalize and calculate weights (b,t,1)
        attention_vec = inputs * attention_weight
        attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec, attention_weight


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self,x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x
    
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class Model(torch.nn.Module):
    def __init__(self, vocab_size1, vocab_size2, hidden_dim1, hidden_dim2, dropout_rate, hidden_channels, out_channels):
        super().__init__()
        self.device = "cuda:0"
        self.MLP_word = Linear(-1, hidden_dim2)
        self.MLP_res = Linear(-1, hidden_dim2)
        self.lstm = LSTM(vocab_size1, hidden_dim1, hidden_dim2, dropout_rate)
        self.lstm_bert = LSTM(vocab_size2, hidden_dim1, hidden_dim2, dropout_rate)
        self.attention = FusionAttention(hidden_dim2)
        self.attention_bert = FusionAttention(hidden_dim2)
        self.gat = GAT(hidden_channels, out_channels)
        self.gcn = GNN(hidden_channels, out_channels)
        self.fusion = Linear(-1,2)
        self.act = nn.ReLU()
        
    def forward(self,data, ids, bert_ids, length, abl):
        doc_feat = self.lstm(ids, length)
        doc_bert_feat = self.lstm_bert(bert_ids, length)
        doc_feat,_ = self.attention(doc_feat)
        doc_bert_feat,_ = self.attention_bert(doc_bert_feat)
        
        word_feat = self.act(self.MLP_word(data.x_dict['word']))
        resp_feat = self.act(self.MLP_res(data.x_dict['responser']))
        data.x_dict['document'] = doc_feat
        data.x_dict['word'] = word_feat
        data.x_dict['responser'] = resp_feat
        
        
        if abl == 0: # original
            graph_feature = self.gat(data.x_dict, data.edge_index_dict)
            feature = torch.cat([doc_feat,graph_feature['document'], doc_bert_feat],dim = -1)
            
        elif abl == 1: # no bert
            graph_feature = self.gat(data.x_dict, data.edge_index_dict)
            feature = torch.cat([doc_feat,graph_feature['document']],dim = -1)
            
        elif abl == 2: # no general
            graph_feature = self.gat(data.x_dict, data.edge_index_dict)
            feature = torch.cat([graph_feature['document'], doc_bert_feat],dim = -1)
            
        elif abl == 3: # no graph
            feature = torch.cat([doc_feat, doc_bert_feat],dim = -1)
            
        elif abl == 4: # only lstm +att
            feature = doc_feat
            
        elif abl == 5: # only gat
            graph_feature = self.gat(data.x_dict, data.edge_index_dict)
            feature = graph_feature['document']
            
        elif abl == 6: # only gcn
            graph_feature = self.gcn(data.x_dict, data.edge_index_dict)
            feature = graph_feature['document']
        
        feature = self.fusion(feature)
        
        return feature
    
    def init_gnn(self,data,abl):
        if abl == 6:
            with torch.no_grad():
                self.gcn = to_hetero(self.gcn, data.metadata(), aggr="sum").to(self.device)
                out = self.gcn(data.x_dict, data.edge_index_dict)
        else:
            with torch.no_grad():
                self.gat = to_hetero(self.gat, data.metadata(), aggr="sum").to(self.device)
                out = self.gat(data.x_dict, data.edge_index_dict)
            
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
seed = [1,34,56,78,46,89,90,27,989,1000]

for seeds in seed:    
    same_seeds(seeds)
    
    model = Model(vocab_size1 = len(distinct_word), vocab_size2 = len_bert_vocab, hidden_dim1 = h1 , hidden_dim2 = h2 , dropout_rate = 0.2, hidden_channels=hc, out_channels=oc).to(device)

    model.init_gnn(data, ablation)

    optimizer = torch.optim.Adam(model.parameters(), lr = lrs, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    num_epoch = 1000
    max_acc = 0
    target = data.y_dict['document']

    for epoch in range(num_epoch):
        model.train()
        optimizer.zero_grad()

        logits = model(data, ids, bert_ids, length, ablation)

        tr_loss = criterion(logits[train_id], target[train_id])  
        val_loss = criterion(logits[val_id], target[val_id]) 


        tr_loss.backward()
        optimizer.step()
        tr_pred = torch.max(logits[train_id].data, 1)[1].cpu().numpy()
        tr_tar = target[train_id].data.cpu().numpy()
        train_acc = accuracy_score(tr_pred, tr_tar)

        scheduler.step(tr_loss)  

        val_pred = torch.max(logits[val_id].data, 1)[1].cpu().numpy()
        val_tar = target[val_id].data.cpu().numpy()
        val_acc = accuracy_score(val_pred, val_tar)

        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model,"best_model.pt")

    best_model = torch.load("best_model.pt")

    metric_collection = MetricCollection([
        Accuracy(),
        F1(num_classes=2, average='macro'),
        Precision(num_classes=2, average='macro'),
        Recall(num_classes=2, average='macro')]).to(device)

    test_logits = best_model(data, ids, bert_ids, length, ablation)
    test_pred = torch.max(test_logits[test_id].data, 1)[1].to(device)
    test_tar = target[test_id].data.to(device)
    metrics = metric_collection(test_pred, test_tar)

    print(metrics)


