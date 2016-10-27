from datetime import datetime
import sys

def sgd(model, X_train, Y_train, lr=0.01, decay=0.9, nepoch=100, eval_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # evaluate loss
        if not epoch%eval_after and epoch:
            loss = model.loss(X_train, Y_train)
            print('\nLoss after {0} iterations : {1}'.format(epoch+1,loss))
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # adjust learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                lr *= 0.5
                print('Learning Rate set to {}'.format(lr))
            sys.stdout.flush()

        for i in range(len(Y_train)):
            # sgd step
            model.sgd_step(X_train[i],Y_train[i], lr=lr, decay=decay)
            num_examples_seen += 1
            sys.stdout.write('\r{0}% complete.'.format( (epoch*len(Y_train) +i+1)/(nepoch * len(Y_train))*100 ))
