import numpy as np
import theano
import theano.tensor as T


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum(axis=0)) # only difference


class RNNNumpy:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # init params
        #   uniform -> (low, high, shape)
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), 
                [hidden_dim, word_dim])
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), 
                [word_dim, hidden_dim])
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), 
                [hidden_dim, hidden_dim])

    def forward(self, x, batch_size):
        # find number of time steps in x
        T = len(x)
        # maintain all hidden states, including initial (0)
        s = np.zeros( [T+1, self.hidden_dim, batch_size] )
        # save output at each time step
        o = np.zeros([T, self.word_dim, batch_size])
        # for each time step
        for t in range(T):
            # select column from U based on index in x
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            # s[t-1] -> initial hidden state is kept as last element of s : s[-1]
            o[t] = softmax(self.V.dot(s[t]))

        return [o,s]
    
    def predict(self,x):
        o, s = self.forward(x, x.shape[-1])
        return np.argmax(o, axis=1)

    def loss(self,x,y):
        # total number of predictions (consider every word)
        N = np.sum( [ len(y_i) for y_i in y ] )
        # divide by total loss
        return self._loss(x,y)/N

    # calculate CCE
    def _loss(self,x,y):
        L = 0
        # run examples one by one
        for i in np.arange(len(y)):
            o, s = self.forward(x[i])
            # o.shape : [10x8000]
            # y[i].shape : [10x1]
            # correct_word_predictions : [10x1] -> probabilities of correct labels
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            L += -np.sum(np.log(correct_word_predictions))
        return L

    # bptt
    def bptt(self, x, y):
        T = len(y)
        # forward propagation
        o, s = self.forward(x)
        # accumulate gradients
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        
        delta_o = o
        delta_o[np.arange(len(y)),y] -= 1
        # for each output, propagate error backwards -> [::-1]
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # initial delta
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # BPTT
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    # sgd
    def sgd_step(self, x, y, lr):
        # calculate gradients
        dLdU, dLdV, dLdW = self.bptt(x,y)
        # update parms
        self.U -= lr * dLdU
        self.V -= lr * dLdV
        self.W -= lr * dLdW



class RNNTheano:

    def __init__(self, word_dim, hidden_dim=50, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # init params
        #   uniform -> (low, high, shape)
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), 
                [hidden_dim, word_dim])
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), 
                [word_dim, hidden_dim])
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), 
                [hidden_dim, hidden_dim])
        # symbolic inputs
        self.U = theano.shared(value=U.astype(theano.config.floatX),name='U')
        self.V = theano.shared(value=V.astype(theano.config.floatX),name='V')
        self.W = theano.shared(value=W.astype(theano.config.floatX),name='W')
        # store theano graph here
        self.theano = {}
        # function -> build graph
        self.__theano_build__()

    # build graph here
    def __theano_build__(self):
        # params
        U,V,W = self.U,self.V,self.W
        # input symbols
        x = T.ivector('x')
        y = T.ivector('y')

        # forward propagation step
        def forward(x_t, s_t_prev,U,V,W):
            s_t = T.tanh( U[:,x_t] + W.dot(s_t_prev) )
            o_t = T.nnet.softmax(V.dot(s_t))
            # softmax returns an array of one element
            return [o_t[0], s_t]

        # scan loop
        [o,s], updates = theano.scan( fn=forward, sequences=x, non_sequences=[U,V,W], outputs_info=[ None, {'initial' : np.zeros(self.hidden_dim)} ],
                truncate_gradient=self.bptt_truncate, strict=True )

        # prediction
        prediction = T.argmax(o, axis=1)
        # loss
        cce = T.sum(T.nnet.categorical_crossentropy(o,y))
        # gradients
        dU = T.grad(cce,U)
        dV = T.grad(cce,V)
        dW = T.grad(cce,W)

        # bind functions to object
        self.forward = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.get_cce = theano.function([x,y], cce)
        self.bptt = theano.function([x,y], [dU,dV,dW])

        # sgd
        lr = T.scalar('lr')
        self.sgd_step = theano.function([x,y,lr], [], updates = [ (self.U, self.U - lr*dU), (self.V, self.V - lr*dV), (self.W, self.W - lr*dW) ])

    # calculate loss
    def _loss(self,x,y):
        return np.sum( [ self.get_cce(xi,yi) for xi,yi in zip(x,y) ])

    def loss(self,x,y):
        num_words = np.sum([ len(yi) for yi in y ])
        return self._loss(x,y)/num_words


class GRUTheano:

    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # init params
        #   uniform -> (low, high, shape)

        # Word Embedding
        #   Input  : [d] vector (word_dim = 8000)
        #   Output : [h] vector (hidden_dim = 128)
        #   Shape : [dxh] matrix
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), 
                [hidden_dim, word_dim])
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), 
                [3, hidden_dim, hidden_dim])
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), 
                [3, word_dim, hidden_dim])
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), 
                [3, hidden_dim, hidden_dim])
        # bias 
        b = np.zeros([3,hidden_dim])
        c = np.zeros(word_dim)

        # symbolic inputs
        self.E = theano.shared(value=E.astype(theano.config.floatX),name='E')
        self.U = theano.shared(value=U.astype(theano.config.floatX),name='U')
        self.V = theano.shared(value=V.astype(theano.config.floatX),name='V')
        self.W = theano.shared(value=W.astype(theano.config.floatX),name='W')
        self.b = theano.shared(value=b.astype(theano.config.floatX),name='b')
        self.c = theano.shared(value=c.astype(theano.config.floatX),name='c')

        # Cache parameters for RMSProp
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        # store theano graph here
        self.theano = {}
        # function -> build graph
        self.__theano_build__()

    # build graph here
    def __theano_build__(self):
        # params
        E,U,V,W,b,c = self.E, self.U, self.V, self.W, self.b, self.c
        # input symbols
        x = T.ivector('x')
        y = T.ivector('y')

        # forward propagation step
        def forward(x_t, s_t_prev):
            # s_t = T.tanh( U[:,x_t] + W.dot(s_t_prev) )
            # TODO : Replace with GRU

            # Word embedding
            x_e = E[:, x_t] # select column

            # GRU layer
            z_t = T.nnet.hard_sigmoid( U[0].dot(x_e) + W[0].dot(s_t_prev) + b[0] ) # update gate
            r_t = T.nnet.hard_sigmoid( U[1].dot(x_e) + W[1].dot(s_t_prev) + b[1] ) # reset gate
            c_t = T.tanh( U[2].dot(x_e) + W[2].dot(s_t_prev * r_t) + b[2] ) # combine memory with current input
            s_t = (T.ones_like(z_t)-z_t)*c_t + z_t*s_t_prev # internal state

            o_t = T.nnet.softmax(V.dot(s_t) + c)[0] # output
            # softmax returns an array of one element
            return [o_t, s_t]

        # scan loop
        [o,s], updates = theano.scan( fn=forward, sequences=x, outputs_info=[ None, {'initial' : np.zeros(self.hidden_dim)} ],
                truncate_gradient=self.bptt_truncate)

        # prediction
        prediction = T.argmax(o, axis=1)
        # loss
        cce = T.sum(T.nnet.categorical_crossentropy(o,y))
        # gradients
        dE = T.grad(cce,E)
        dU = T.grad(cce,U)
        dV = T.grad(cce,V)
        dW = T.grad(cce,W)
        db = T.grad(cce,b)
        dc = T.grad(cce,c)

        # bind functions to object
        self.forward = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.get_cce = theano.function([x,y], cce)
        self.bptt = theano.function([x,y], [dE,dU,dV,dW,db,dc])

        # sgd
        lr = T.scalar('lr')
        #self.sgd_step = theano.function([x,y,lr], [], updates = [ (self.E, self.E - lr*dE), (self.U, self.U - lr*dU), (self.V, self.V - lr*dV), (self.W, self.W - lr*dW) ])

        # Use RMSProp
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = decay*self.mE + (1-decay)*dE**2
        mU = decay*self.mU + (1-decay)*dU**2
        mW = decay*self.mW + (1-decay)*dW**2
        mV = decay*self.mV + (1-decay)*dV**2
        mb = decay*self.mb + (1-decay)*db**2
        mc = decay*self.mc + (1-decay)*dc**2

        # step
        self.sgd_step = theano.function( [x,y,lr, decay], 
                [],
                updates = [ (E, E - lr*dE / T.sqrt(mE + 1e-6)),
                    (U, U - lr*dU / T.sqrt(mU + 1e-6)),
                    (V, V - lr*dV / T.sqrt(mV + 1e-6)),
                    (W, W - lr*dW / T.sqrt(mW + 1e-6)),
                    (b, b - lr*db / T.sqrt(mb + 1e-6)),
                    (c, c - lr*dc / T.sqrt(mc + 1e-6)),
                    (self.mE, mE),
                    (self.mU, mU),
                    (self.mV, mV),
                    (self.mW, mW),
                    (self.mb, mb),
                    (self.mc, mc)
                    ])


    # calculate loss
    def _loss(self,x,y):
        return np.sum( [ self.get_cce(xi,yi) for xi,yi in zip(x,y) ])

    def loss(self,x,y):
        num_words = np.sum([ len(yi) for yi in y ])
        return self._loss(x,y)/num_words

