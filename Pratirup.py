import Tanitra
import cupy as cp

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

    def learn(self,X,y,optimizer = 'Gradient Descent',epochs = 1000,lr = 0.01,tol = 0.00001):
        if optimizer == 'Gradient Descent':
            loss = 0
            for _ in range(epochs):
                prev_loss = loss
                loss = 0
                for i in range(Tanitra.length(X)):
                    y_pred = self.estimate(X[i])
                    loss += Tanitra.to_cons(Tanitra.mean(Tanitra.square(y_pred - y[i])) / Tanitra.length(X) )
                    req_loss = Tanitra.square(y_pred - y[i])/ Tanitra.length(X)
                    req_loss.backward()
                    for j in self.layers:
                        for k in j.params:
                            j.params[k].data = j.params[k].data - j.params[k].grad*lr/Tanitra.length(X)
                if prev_loss - loss < tol and _ != 0:
                    break
                if not (prev_loss - loss < tol and _ != 0) and _ == epochs - 1:
                    raise RuntimeWarning("Error did not converge. Please consider increasing number of epochs.")
                print("Epoch no. ",_)
                print("  loss is ",loss)
                print(" ")

class ShabdAyamahPratirup:

    def __init__(self, embedding_dimension,activation):
        self.vocabulary = None
        self.token_list = {}
        self.embedding_dimension = embedding_dimension
        self.activation = activation
        self.sentences = None
        self.params = {}
        self.sentence_indices = None

    def sentence2indices(self,sentences, token_identification = 'separate by space', token_list = None):
        vocabulary = 0
        sentence_indices = Tanitra.Tanitra([])
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
        self.vocabulary = vocabulary
        self.sentence_indices = sentence_indices
        return(sentence_indices)

    def learn(self,sentences):
        self.sentence_indices.append(sentences)
        training_data_x = Tanitra.Tanitra([])
        training_data_y = Tanitra.Tanitra([])
        for i in self.sentence_indices:
            for j in range(len(i)):
                train_x = cp.zeros(self.vocabulary)
                train_y = cp.zeros(self.vocabulary)
                train_x[j] = 1
                train_y[i] = cp.ones_like(train_y[:j])/len(i)-1
                train_y[j] = 0
                training_data_x = training_data_x.append(train_x)
                training_data_x = training_data_x.append(train_y)
                training_data_y = training_data_y.append(train_y)
                training_data_y = training_data_y.append(train_x)

model = ShabdAyamahPratirup(123,'relu')
a = model.sentence2indices([
    "at"
])
print('vocabulary:')
print(model.sentences)
print(model.vocabulary)
print(model.token_list)
print(model.sentence_indices)