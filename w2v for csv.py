# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:22:19 2020

@author: Home
"""

# from torchtext import data
# import en_core_web_sm
# nlp = en_core_web_sm.load()

# def tokenizer(text): # create a tokenizer function
#     return [tok.text for tok in nlp.tokenizer(text)]

# TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
# TEXT_1 = data.Field(sequential=True, tokenize=tokenizer, lower=True)


# train = data.TabularDataset.splits(
#         path ='C:/Users/Home/Desktop/matching3.csv', format='csv',
#         fields=[('Text', TEXT), ('Text_1', TEXT_1)],skip_header = True)

# TEXT.build_vocab(train)
# TEXT_1.build_vocab(train)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gensim

import torch 
import torch.nn as nn

import torch.utils.data as data
import torch.optim as optim
import re
from string import punctuation

training_data = pd.read_csv('C:/Users/Home/Desktop/more records.csv', dtype = {"job" : "string", "resume" : "string"})



# //////////////////////////////////////////////////////////////////
# ///////////////////////CLEANING///////////////////////////////////
# ////////////////////////////////////////////////////////////////

print(training_data.res_des.head)
clear_punct_regex = "[" + punctuation + "\d\r\n]"
training_data['res_des'] = training_data['res_des'].str.replace(clear_punct_regex, "").str.lower()
training_data['job_des'] = training_data['job_des'].str.replace(clear_punct_regex, "").str.lower()
print(training_data.res_des.head)


# puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
#  '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
#  '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
#  '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
#  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

# def clean_text(x):
#     x = str(x)
#     for punct in puncts:
#         if punct in x:
#             x = x.replace(punct, f' {punct} ')
#     return x


# def clean_numbers(x):
#     x = str(x)
#     if bool(re.search(r'\d', x)):
#         x = re.sub('[0-9]{5,}', '#####', x)
#         x = re.sub('[0-9]{4}', '####', x)
#         x = re.sub('[0-9]{3}', '###', x)
#         x = re.sub('[0-9]{2}', '##', x)
#     return x

# def words(text):
#     text=str(text)
#     return re.findall(r'\w+', text.lower())

# print(training_data.job_des.head)
# clean_text(training_data)
# clean_numbers(training_data)

# words(training_data.job_des)
# print(training_data.job_des.head)



# ///////////////////////////////////////////////////////////////////////
# ////////////////////////////SPLITTING/////////////////////////////////
# //////////////////////////////////////////////////////////////////////

training_data_copy = training_data.copy()
training_data = training_data_copy.sample(frac=0.75, random_state=0)
testing_data = training_data_copy.drop(training_data.index)

# msk = np.random.rand(len(training_data)) <= 0.8
# training_data = training_data[msk]
# testing_data = training_data[~msk]

print(len(training_data))
print(len(testing_data))


labels = training_data.label.values
resumes = training_data.resume.values


# print(resumes)
# print(labels[:])
target=torch.tensor(labels)


print(target)


# building a dictionary
# label_map = {
#     label:idx
#     for idx,label in enumerate(labels)
# }

# print(label_map)


# target=pd.unique(training_data.label)
# print(target)
# target= torch.tensor(target)
# print(target.shape)

# target=('0','1','2','3','4','5','6','7','8','9','10','11','12','13')
# target=torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,10,4,13])
# print(training_data.head())
# print(training_data)



(
    training_data
    .job_des
    .apply(lambda x: x.split(" "))
    .head()
    .values
)
# print(training_data)


training_corpus = training_data.job_des.apply(lambda x: x.split(" "))
# print(training_corpus)

model = gensim.models.Word2Vec(sentences=training_corpus,size=100, alpha=0.025, window=5, min_count=1, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), max_final_vocab=None )
# print(model)
# 
model.wv.vocab
# print(model.wv.vocab)
print(len(model.wv.vocab))
# print(model.wv['JD'])
# print(model.wv.most_similar("Manager"))

model.wv.vectors
# print(len(model.wv.vectors))



# class FirstNeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super().__init__()
#         self.layer1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.layer2 = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, data_input):
#         output_layer1 = self.layer1(data_input)
#         output_layer1 = self.relu(output_layer1)
#         output_layer2 = self.layer2(output_layer1)
#         return output_layer2




class FirstNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.embedding_layer = nn.EmbeddingBag.from_pretrained(word_vectors_for_training, mode="mean")
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, data_input):
        embedded_data_input = self.embedding_layer(data_input)
        # print(embedded_data_input.size())
        output_layer1 = self.layer1(embedded_data_input)
        # print(output_layer1.size())
        output_layer1 = self.relu(output_layer1)
        # print(output_layer1.size())
        output_layer2 = self.layer2(output_layer1)
        # return the predictions but drop the axis
        # print(output_layer2.size())
        return output_layer2
    
    
    
# first_neural_network = FirstNeuralNetwork(300, 100, 2)
# print(first_neural_network)



sentence_length = training_data.job_des.apply(lambda x: x.split(" ")).apply(len).max()
# print(sentence_length)


word_map = {
    word:idx
    for idx,word in enumerate(model.wv.vocab, start=2)
}
# print(word_map)

(
    training_data
    .job_des
    .apply(lambda x: x.split(" "))
)[0]

training_sentence_data = (
    training_data
    .job_des
    .apply(lambda x: list(map(lambda word: word_map[word] if word in word_map else 1, x.split(" "))))
)
# print(training_sentence_data[1])


training_sentence_data = (
        list(
            map(
                lambda x: pd.np.pad(x, (0, sentence_length-len(x))), 
                training_sentence_data
            )
        )
)
# print(training_sentence_data[0])


# torch.LongTensor(training_sentence_data[0])

training_sentence_data = (
    list(
        map(
            lambda x: torch.LongTensor(x), 
            training_sentence_data
        )
    )
)
# print(training_sentence_data[1])

pd.np.random.seed(100)

print(model.wv.vectors.shape)

word_vectors_for_training = pd.np.insert(
    model.wv.vectors,   
    0, 
    pd.np.random.uniform(model.wv.vectors.min(),model.wv.vectors.max(),100),
    axis=0
)

word_vectors_for_training = pd.np.insert(
    word_vectors_for_training,   
    0, 
    pd.np.zeros(100),
    axis=0
)
word_vectors_for_training = torch.FloatTensor(word_vectors_for_training)
print(word_vectors_for_training.shape)




# finalize the training data
training_sentence_data = torch.stack(training_sentence_data)
print(training_sentence_data.shape)
# Define the network
first_neural_network = FirstNeuralNetwork(100, 50, 42)

# ///////////////////////////////////////////////////////////////////////
# /////////////////////DO THE SAME FOR TARGET///////////////////////////
# //////////////////////////////////////////////////////////////////////
# (
#     training_data
#     .res_des
#     .apply(lambda x: x.split(" "))
#     .head()
#     .values
# )


# training_corpus2 = training_data.res_des.apply(lambda x: x.split(" "))
# # print(training_corpus2)


# model2 = gensim.models.Word2Vec(sentences=training_corpus2,size=100, alpha=0.025, window=5, min_count=1, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), max_final_vocab=None )
# # print(model2)


# model2.wv.vocab
# # print(model2.wv.vocab)
# # print(len(model2.wv.vocab))

# model2.wv.vectors
# # print(model2.wv.vectors)

# # //////////
# sentence_length2 = training_data.res_des.apply(lambda x: x.split(" ")).apply(len).max()
# # print(sentence_length2)


# word_map2 = {
#     word:idx
#     for idx,word in enumerate(model2.wv.vocab, start=2)
# }
# # print(word_map2)
# # ////////////
# (
#     training_data
#     .res_des
#     .apply(lambda x: x.split(" "))
# )[0]

# training_sentence_data2 = (
#     training_data
#     .res_des
#     .apply(lambda x: list(map(lambda word: word_map2[word] if word in word_map2 else 1, x.split(" "))))
# )
# # print(training_sentence_data2[0])

# training_sentence_data2 = (
#     list(
#         map(
#             lambda x: pd.np.pad(x, (0, sentence_length2-len(x))), 
#             training_sentence_data2
#         )
#     )
# )
# # print(training_sentence_data2[0])

# training_sentence_data2 = (
#     list(
#         map(
#             lambda x: torch.LongTensor(x), 
#             training_sentence_data2
#         )
#     )
# )

# # print(training_sentence_data2[0])

# pd.np.random.seed(100)

# model2.wv.vectors.shape

# word_vectors_for_training2 = pd.np.insert(
#     model2.wv.vectors,   
#     0, 
#     pd.np.random.uniform(model2.wv.vectors.min(),model2.wv.vectors.max(),100),
#     axis=0
# )

# # print(word_vectors_for_training2)


# word_vectors_for_training2 = pd.np.insert(
#     word_vectors_for_training2,   
#     0, 
#     pd.np.zeros(100),
#     axis=0
# )
# # print(word_vectors_for_training2)

# word_vectors_for_training2 = torch.FloatTensor(word_vectors_for_training2)
# print(word_vectors_for_training2.shape)

# # finalize the training data
# training_sentence_data2 = torch.stack(training_sentence_data2)

# print(training_sentence_data2.shape)
# # ////////////////////////////////////////////////////////////////////////
# # ////////////////////////////////////////////////////////////////////////
# # /////////////////////////////////////////////////////////////////////////

# # target = target.unsqueeze(1),
# # training_sentence_data2 = training_sentence_data2.view(-1,1,1,1)
# training_sentence_data = training_sentence_data.view(-1,8)
# # print(training_sentence_data.size(0))
# # print(training_sentence_data2.size(0))
# print(training_sentence_data.size(0))
# print(target.size(0))


# dataset = data.TabularDataset
dataset = data.TensorDataset(training_sentence_data,target)
# print(training_sentence_data.size())
# print(training_sentence_data2.size())

# print(training_sentence_data2.size())
# training_sentence_data2=training_sentence_data2.reshape(17)
# print(training_sentence_data2.size())
dataloader = data.DataLoader(dataset, batch_size=2)
# print(training_sentence_data.size())
# print(training_sentence_data2.size())

optimizer = optim.Adam(first_neural_network.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()



# loss = nn.BCEWithLogitsLoss()
# in_mat = torch.randn(3, requires_grad=True)
# tar_mat = torch.randn(3, requires_grad=True)
# output = loss(in_mat, tar_mat)
# # print(training_sentence_data.size())
# # print(training_sentence_data2.size())

# print(in_mat)
# print(tar_mat)
# print(output)


#progress bar
import tqdm

# number of epochs
for n in tqdm.tqdm(range(700)):
    avg_loss = []

    for batch in dataloader:
        print(batch[0].size())
        print(batch[1].size())
        
        # batch [1] - the label for each sentence or Y
        # print(batch[1])
        # for every back pass you need to zero out the optimizer
        # less you get residual gradients
        optimizer.zero_grad()
        # pass the model into the batch
        # this line is the same as calling first_neural_network.foward(batch)
        # yay shortcuts
        # print(batch[1])
        output = first_neural_network(batch[0])
        print(output)
        # print(batch[0].size())
        # print(batch[1].size())
        # Calculate the loss function
        # print(output, batch[1])
        
        # batch[1] = batch[1].type_as(output)
        # batch[1] = nn.Sigmoid()(batch[1])
        loss = loss_fn(output, batch[1])
       
        # Save the loss for each epoch
        avg_loss.append(loss.item())
        
        # Tell pytorch to calculate the gradient
        loss.backward()
        
        # tell pytorch to pass the gradients back into the model
        optimizer.step()
        
    print(pd.np.mean(avg_loss))
    
    
    
# ///////////////////////////////////////////////////////////////////
# //////////////////////////TESTING//////////////////////////////////
# //////////////////////////////////////////////////////////////////
# load testing data
# testing_data = pd.read_csv('C:/Users/Home/Desktop/tests2.csv')
print(testing_data.head())
    
testing_sentence_data = (
    testing_data
    .job_des
    .apply(lambda x: list(map(lambda word: word_map[word] if word in word_map else 1, x.split(" "))))
)
# print(testing_sentence_data[0])
    
testing_sentence_data = (
    list(
        map(
            lambda x: pd.np.pad(x, (0, sentence_length-len(x))), 
            testing_sentence_data
        )
    )
)
# print(testing_sentence_data[0])

testing_sentence_data = (
    list(
        map(
            lambda x: torch.LongTensor(x), 
            testing_sentence_data
        )
    )
)


# print(testing_sentence_data[0])

# finalize the testing data
testing_sentence_data = torch.stack(testing_sentence_data)
print(testing_sentence_data.shape)

test_dataset = data.TensorDataset(testing_sentence_data)
test_dataloader = data.DataLoader(test_dataset, batch_size=256)


prediction_data = testing_data.loc[:,["id","job"]].copy()
print(prediction_data.size)
lowest_idx=0
for t_batch in test_dataloader:
        # batch [0] - the sentence data or X
        
        
        # pass the model into the batch
        # this line is the same as calling first_neural_network.foward(batch)
        # yay shortcuts
        output = nn.functional.softmax(first_neural_network(t_batch[0]))
        print(output.detach().numpy())
        bsize = len(output)
        print(bsize)
        max_index_col = np.argmax(output.detach().numpy(), axis=0)
        max_index_row = np.argmax(output.detach().numpy(), axis=1)
        # range(lowest_idx,lowest_idx+bsize)
        prediction_data[lowest_idx,"prob_target"] = max_index_row
        # lowest_idx += bsize
        lowest_idx+=1
print(prediction_data)

