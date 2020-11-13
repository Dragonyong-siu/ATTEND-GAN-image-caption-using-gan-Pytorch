#6) Caption_Discriminator
 #6.1) using gru to get hidden_state and then get score between [0, 1]

class Caption_Discriminator(nn.Module):
  def __init__(self):
    super(Caption_Discriminator, self).__init__()

    # gpt2_embedding
    self.gpt2_model = GPT2_Model
    self.gpt2_wte = self.gpt2_model.wte

    # dimensions
    self.gpt2_dim = 768
    self.gru_in = 768
    self.gru_hidden = 1024
    self.layers_num = 2
    self.directions_num = 2
    self.p = 0.1
    self.batch = 2
    self.seq_len = 16
    self.linear_in = self.layers_num * self.directions_num * self.gru_hidden

    # using gru to get hidden_state
    self.gru = nn.GRU(self.gru_in, 
                      self.gru_hidden, 
                      num_layers = self.layers_num, 
                      bidirectional = True,
                      dropout = self.p)
    
    # do linear_regression : Wx + b
    self.linear_gru2logit = nn.Linear(self.linear_in, self.gru_hidden)  
    self.linear_logit2score = nn.Linear(self.gru_hidden, 1)
    self.dropout = nn.Dropout(p = self.p)

    # activation_function
    self.relu = nn.ReLU(inplace = True)
    self.sigmoid = nn.Sigmoid()

  def forward(self, caption):  

    # caption
    caption = self.gpt2_wte(caption) #torch.Size([2, 16, 768])
    caption = caption.permute(1, 0, 2) #torch.Size([16, 2, 768])

    # hidden
    hidden = torch.zeros(self.layers_num * self.directions_num,
                         self.batch,
                         self.gru_hidden) #torch.Size([4, 2, 1024])
    hidden = hidden.to(device)
    _, hidden = self.gru(caption, hidden) #torch.Size([4, 2, 1024])
    hidden = hidden.permute(1, 0, 2) #torch.Size([2, 4, 1024])
    hidden = hidden.reshape(self.batch, self.linear_in) #(2, 4096)

    # logit
    logit = self.linear_gru2logit(hidden)
    logit = self.relu(logit)
    logit = self.dropout(logit) #torch.Size([2, 1024])

    # score
    score = self.linear_logit2score(logit) #torch.Size([2, 1024])
    score = self.sigmoid(score)

    return score

ATTEND_GAN_Discriminator = Caption_Discriminator().to(device)
