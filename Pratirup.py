import Tanitra
import cupy as cp
import matplotlib.pyplot as plt
import Parata
import numpy as np

class AnukramikPratirup:

    def __init__(self,layers = None):
        self.layers = []
        if not layers is None:
            for i in layers:
                self.add(i)

    def add(self,layer):
        self.layers.append(layer)

    def estimate(self,X):
        activation = X
        for i in self.layers:
            activation = i.forward(activation)
        return activation

    def learn(self,X,y,optimizer = 'Gradient Descent',epochs = 1000,lr = 0.1,tol = 0.00001):
        if optimizer == 'Gradient Descent':
            loss = 0
            epoch = []
            loss_a = []
            for _ in range(epochs):
                prev_loss = loss
                loss = 0
                for i in range(Tanitra.length(X)):
                    print('yes')
                    y_pred = self.estimate(X[i])
                    loss += Tanitra.to_cons(Tanitra.mean(Tanitra.square(y_pred - y[i])) / Tanitra.length(X) )
                    req_loss = Tanitra.square(y_pred - y[i])/ Tanitra.length(X)
                    req_loss.backward()
                    for j in self.layers:
                        for k in j.params:
                            print('yes')
                            j.params[k].data = j.params[k].data - j.params[k].grad*lr/Tanitra.length(X)
                    req_loss.grad_0()
                if prev_loss - loss < tol and _ != 0:
                    break
                if not (prev_loss - loss < tol and _ != 0) and _ == epochs - 1:
                    print("\n\nError did not converge. Please consider increasing number of epochs.")
                print("Epoch no. ",_)
                epoch.append(_)
                loss_a.append(float(loss))
                print("  loss is ",loss)
                print(" ")
            plt.plot(epoch, loss_a)
            plt.show()


class ShabdAyamahPratirup:

    def __init__(self, embedding_dimension,method):
        self.vocabulary = 0
        self.token_list = {}
        self.embedding_dimension = embedding_dimension
        self.method = method
        self.sentences = []
        self.sentence_indices = []
        self.params = {}
        self.layers = [Parata.GuptaParata(self.embedding_dimension, 'linear',)]
        self.params_initialized = False
        self.grad = {}

    def sentence2indices(self,sentences, token_identification = 'separate by space', token_list = None):
        vocabulary = 0
        sentence_indices = []
        if token_identification == 'separate by space':
            current_token = ''
            for i in range(len(sentences)):
                sentence_indices_ith = []
                for j in sentences[i]:
                    if j.isupper():
                        j = j.lower()
                    if (j != ' ' and not 33 <= ord(j) <= 47 or 58 <= ord(j) <= 64 or 91 <= ord(j) <= 96 or 123 <=
                            ord(j) <= 126):
                        current_token += j
                    elif j == ' ':
                        if not current_token in self.token_list:
                            self.token_list[current_token] = vocabulary
                            sentence_indices_ith.append(vocabulary)
                            vocabulary += 1
                        else:
                            sentence_indices_ith.append(self.token_list[current_token])
                        current_token = ''
                if not current_token in self.token_list:
                    self.token_list[current_token] = vocabulary
                    sentence_indices_ith.append(vocabulary)
                    vocabulary += 1
                else:
                    sentence_indices_ith.append(self.token_list[current_token])
                current_token = ''
                sentence_indices.append(sentence_indices_ith)
        self.vocabulary += vocabulary
        for i in sentence_indices:
            self.sentence_indices.append(i)
        return(sentence_indices)

    def forward_fake(self,x):
        if not self.params_initialized:
            self.params = {'embeddings':  Tanitra.Tanitra(cp.random.randn(self.vocabulary, self.embedding_dimension)
                                                          * (1.0 / cp.sqrt(self.vocabulary))),
                           'weight_dash': Tanitra.Tanitra(cp.random.randn(self.embedding_dimension, self.vocabulary))
                                          / cp.sqrt(self.vocabulary/2)}
            self.grad = {'embeddings':  Tanitra.Tanitra(cp.zeros((self.vocabulary, self.embedding_dimension))),
                         'weight_dash': Tanitra.Tanitra(cp.zeros((self.embedding_dimension, self.vocabulary)))}
            self.params_initialized = True
        if not isinstance(x,Tanitra.Tanitra):
            x = Tanitra.Tanitra(x)
        y = x@self.params['embeddings']
        y1 = y@self.params['weight_dash']
        y2 = Tanitra.softmax(y1)
        return y2

    def forward(self,x):
        if not self.params_initialized:
            self.params = { 'embeddings':Tanitra.Tanitra(cp.random.randn(self.vocabulary, self.embedding_dimension)
                                                         * (1.0 / cp.sqrt(self.vocabulary))),
                           'weight_dash': Tanitra.Tanitra(cp.random.randn(self.embedding_dimension, self.vocabulary))
                                          / cp.sqrt(self.vocabulary/2)}
            self.params_initialized = True
        return self.params['embeddings'][x]

    def learn(self,epochs,lr,tol,window_size,sentences = None):
        if sentences is not None:
            for i in sentences:
                self.sentence_indices.append(i)
        self.layers.append(Parata.GuptaParata(self.vocabulary,'softmax'))
        training_data_x = []
        training_data_y = []
        epoch = []
        loss_a = []
        for sentence in self.sentence_indices:
            for j in range(len(sentence) - window_size+1):
                window = sentence[j:j + window_size + 1]
                for target_pos in range(len(window)):
                    target = window[target_pos]
                    train_x = cp.zeros(self.vocabulary)
                    train_x[target] = 1
                    train_y = cp.zeros(self.vocabulary)
                    for context_pos in range(len(window)):
                        if context_pos != target_pos:
                            context = window[context_pos]
                            train_y[context] = 1
                    training_data_x.append(train_x)
                    training_data_y.append(train_y/4)
        print(training_data_x)
        print(training_data_y)
        loss = 0
        for _ in range(epochs):
            prev_loss = loss
            loss = 0
            for i in range(len(training_data_x)):
                y_pred = self.forward_fake(training_data_x[i])
                loss += Tanitra.to_cons(Tanitra.mean(Tanitra.square(y_pred - training_data_y[i])) /
                                        len(training_data_x))
                req_loss = Tanitra.square(y_pred - training_data_y[i]) / len(training_data_x)
                req_loss.backward()
                for j in self.params:
                    self.grad[j].data += self.params[j].grad/len(training_data_x)
                req_loss.grad_0()
            for j in self.params:
                self.params[j] = self.params[j] - self.grad[j] * lr
            print("Epoch no. ", _)
            epoch.append(_)
            loss_a.append(float(loss))
            print("  loss is ", loss)
            print(" ")
            if prev_loss - loss < tol and _ != 0:
                break
            if not (prev_loss - loss < tol and _ != 0) and _ == epochs - 1:
                print('loss did not converge. please consider increasing the epochs')
        plt.plot(epoch,loss_a)
        plt.show()

if __name__ == '__main__':
    model = ShabdAyamahPratirup(10,'skip gram')

    model.sentence2indices([
    "A king is a powerful male ruler.",
    "A queen is a powerful female ruler.",
    "A man is an adult male human.",
    "A woman is an adult female human.",
    "Males and females belong to different genders.",
    "A queen is married to a king.",
    "A king and a queen rule a kingdom.",
    "A man and a woman are adults.",
    "A ruler can be a man or a woman.",
    "A queen is not a man but a woman."
])
    model.learn(20,1,1e-10,7)

    # Given word embeddings
    embeddings = model.params['embeddings']

    print(embeddings[model.token_list['queen']].data)
    print(embeddings[model.token_list['king']].data-embeddings[model.token_list['man']].data+
          embeddings[model.token_list['woman']].data)

    # Extract embeddings
    faster_vec = embeddings[model.token_list['queen']].data
    slower_vec = (embeddings[model.token_list['king']].data-embeddings[model.token_list['man']].data+
                  embeddings[model.token_list['woman']].data)

    # Compute cosine similarity
    cosine_similarity = cp.dot(faster_vec, slower_vec) / (cp.linalg.norm(faster_vec) * cp.linalg.norm(slower_vec))
    print(cosine_similarity)
