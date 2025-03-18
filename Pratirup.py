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

    def learn(self,X,y,optimizer = 'Gradient Descent',epochs = 1000,lr = 0.1,tol = 0.00001):
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
                    req_loss.grad_0()
                if prev_loss - loss < tol and _ != 0:
                    break
                if not (prev_loss - loss < tol and _ != 0) and _ == epochs - 1:
                    print("\n\nError did not converge. Please consider increasing number of epochs.")
                print("Epoch no. ",_)
                print("  loss is ",loss)
                print(" ")

class ShabdAyamahPratirup:

    def __init__(self, embedding_dimension,activation):
        self.vocabulary = 0
        self.token_list = {}
        self.embedding_dimension = embedding_dimension
        self.activation = activation
        self.sentences = []
        self.sentence_indices = []
        self.params = {}
        self.params_initialized = False

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
                sentence_indices.append(Tanitra.Tanitra(cp.array(sentence_indices_ith)))
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
            self.params_initialized = True
        y = x@self.params['embeddings']
        y = y@self.params['weight_dash']
        y = Tanitra.softmax(y)
        return y

    def forward(self,x):
        if not self.params_initialized:
            self.params = { 'embeddings':Tanitra.Tanitra(cp.random.randn(self.vocabulary, self.embedding_dimension)
                                                         * (1.0 / cp.sqrt(self.vocabulary))),
                           'weight_dash': Tanitra.Tanitra(cp.random.randn(self.embedding_dimension, self.vocabulary))
                                          / cp.sqrt(self.vocabulary/2)}
            self.params_initialized = True
        return self.params['embeddings'][x]

    def learn(self,epochs,lr,tol,sentences):
        self.sentence_indices.append(sentences)
        training_data_x = Tanitra.Tanitra([])
        training_data_y = Tanitra.Tanitra([])
        for i in self.sentence_indices:
            for j in range(Tanitra.length(i)-1):
                train_x = cp.zeros(self.vocabulary)
                train_y = cp.zeros(self.vocabulary)
                train_x[j] = 1
                train_y[j+1] = 1
                training_data_x.append(train_x)
                training_data_y.append(train_y)
                training_data_x.append(train_y)
                training_data_y.append(train_x)
        loss = 0
        for _ in range(epochs):
            prev_loss = loss
            loss = 0
            for i in range(Tanitra.length(training_data_x)):
                y_pred = self.forward_fake(training_data_x[i])
                loss += Tanitra.to_cons(Tanitra.mean(Tanitra.square(y_pred - training_data_y[i])) /
                                        Tanitra.length(training_data_x))
                req_loss = Tanitra.square(y_pred - training_data_y[i]) / Tanitra.length(training_data_x)
                req_loss.backward()
                for j in self.params:
                    cp.clip(self.params[j].grad, -3, 3, out=self.params[j].grad)
                    self.params[j].data = self.params[j].data - self.params[j].grad * lr
                req_loss.grad_0()
            if prev_loss - loss < tol and _ != 0:
                break
            if not (prev_loss - loss < tol and _ != 0) and _ == epochs - 1:
                print('loss did not converge. please consider increasing the epochs')
            print("Epoch no. ", _)
            print("  loss is ", loss)
            print(" ")

if __name__ == '__main__':


    model = ShabdAyamahPratirup(2,'relu')

    model.sentence2indices(["troll is great","gymkata is great"])

    model.learn(1000,1,0.000000001,Tanitra.Tanitra([5,1,2,3]))

    print(model.params['embeddings'].data)