import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable


# We build a simple model with the inputs and one output layer.
class SimpleClassifier(nn.Module):
    def __init__(self,n_in=2,n_hidden=5,n_out=4):
        # super(SimpleClassifier,self).__init__()
        super().__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
         
        self.linear = nn.Sequential(
            nn.Linear(self.n_in, self.n_hidden),
            nn.Linear(self.n_hidden ,self.n_hidden),   # Hidden layer.
            nn.Dropout(p=0.2),
            nn.Linear(self.n_hidden ,self.n_out),
            nn.BatchNorm1d(self.n_out),
            nn.ReLU()
            )
        self.logprob = nn.LogSoftmax(dim=1)                 # -Log(Softmax probability).
    
    def forward(self,x):
        x = self.linear(x)
        x = self.logprob(x)
        return x

    def fit_transform(self, my_loader, criterium, learning_rate=0.1, weight_decay=1e-4):
        """
        Fit the model, returns list of losses and list of predictions in that order
        """


        # We create the mode, the loss function or criterium and the optimizer 
        # that we are going to use to minimize the loss.
        
        # Adam optimizer with learning rate 0.1 and L2 regularization with weight 1e-4.
        optimizer = torch.optim.Adam(self.parameters(), learning_rate, weight_decay=weight_decay)


        loss_array = []
        pred_array = []
        epoch = 0

        # Taining.


        for k, (d, t) in enumerate(my_loader):
            # Definition of inputs as variables for the net.
            # requires_grad is set False because we do not need to compute the 
            # derivative of the inputs.

            d = Variable(d,requires_grad=False)
            t = Variable(t.long(),requires_grad=False)

            # Set gradient to 0.
            optimizer.zero_grad()
            # Feed forward.
            pred = self(d)
            pred_array.append(pred)
            # Loss calculation.
            loss = criterium(pred,t.view(-1))
            
            loss_array.append(loss.item())
            # Gradient calculation.
            loss.backward()
            
            # Print loss every 10 iterations.
            if k%10==0:
                print('Loss {:.4f} at iter {:d}'.format(loss.item(),k))
                
            # Model weight modification based on the optimizer. 
            optimizer.step()
            
            epoch += 1

        return loss_array, pred_array