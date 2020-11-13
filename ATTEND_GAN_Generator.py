#5) ATTEND_GAN_Generator
 #5.1) using lstm for pretrain & train generator
import random

class Caption_Generator(nn.Module):
  def __init__(self):
    super(Caption_Generator, self).__init__()

    # gpt2_embedding
    self.gpt2_model = GPT2_Model
    self.gpt2_wte = self.gpt2_model.wte
    self.tokenizer = GPT2_Tokenizer

    # dimensions
    self.gpt2_hidden = 768
    self.feature_dim = 2048
    self.lstm_hidden = 1024
    self.att_in = 512
    self.att_out = 1
    self.feature_num = 49
    self.caption_length = 16
    self.vocab_size = 50260
    self.batch_size = 2
    self.index_num = 5
    self.value_min = 0.2
    self.p = 0.0
    
    # input_gate
    self.Hi_linear = nn.Linear(self.lstm_hidden, self.lstm_hidden)
    self.Wi_linear = nn.Linear(self.gpt2_hidden, self.lstm_hidden)
    self.Ai_linear = nn.Linear(self.feature_dim, self.lstm_hidden)

    # forget_gate
    self.Hf_linear = nn.Linear(self.lstm_hidden, self.lstm_hidden)
    self.Wf_linear = nn.Linear(self.gpt2_hidden, self.lstm_hidden)
    self.Af_linear = nn.Linear(self.feature_dim, self.lstm_hidden)
    
    # modulation_gate
    self.Hm_linear = nn.Linear(self.lstm_hidden, self.lstm_hidden)
    self.Wm_linear = nn.Linear(self.gpt2_hidden, self.lstm_hidden)
    self.Am_linear = nn.Linear(self.feature_dim, self.lstm_hidden)
    
    # output_gate
    self.Ho_linear = nn.Linear(self.lstm_hidden, self.lstm_hidden)
    self.Wo_linear = nn.Linear(self.gpt2_hidden, self.lstm_hidden)
    self.Ao_linear = nn.Linear(self.feature_dim, self.lstm_hidden)

    # soft_attention
    self.We_linear = nn.Linear(self.att_in, self.att_out)
    self.Wa_linear = nn.Linear(self.feature_dim, self.att_in)
    self.Wh_linear = nn.Linear(self.lstm_hidden, self.att_in)

    # initial_state
    self.cell_linear = nn.Linear(self.feature_dim, self.lstm_hidden)
    self.hidden_linear = nn.Linear(self.feature_dim, self.lstm_hidden)

    # word_prob
    self.Pa_linear = nn.Linear(self.feature_dim, self.vocab_size)
    self.Ph_linear = nn.Linear(self.lstm_hidden, self.vocab_size)
    
    # activation_function
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim = 1)
    self.dropout = nn.Dropout(p = self.p)

    # sampler_first_input
    self.start_word = self.tokenizer.convert_tokens_to_ids('[start]')

  def forward(self, input_feature, input_ids):
    
    # input_feature
    input_feature = input_feature.view(-1, self.feature_num, self.feature_dim) #(2, 49, 2048)

    # input_wte
    input_ids = input_ids.to(device)
    input_wte = self.gpt2_wte(input_ids) #(2, 16, 768)

    # initial_state
    feature_mean = torch.mean(input_feature, dim = 1) 
    cell = self.cell_linear(feature_mean) #(2, 1024)
    hidden = self.hidden_linear(feature_mean) #(2, 1024)
    att_feature, e_t = self.soft_attention(input_feature, hidden) #torch.Size([2, 2048]) torch.Size([2, 49, 1])

    # lstm_cell
    logits = []
    atts = []
    for i in range(self.caption_length):
      cell, hidden = self.lstm(cell, hidden, input_wte[:, i, :], att_feature)
      att_feature, e_t = self.soft_attention(input_feature, hidden)
      aW = self.Pa_linear(att_feature) 
      hW = self.Ph_linear(hidden) 
      logit = aW + hW
      logit = self.dropout(logit)
      logit = logit.unsqueeze(1)
      e_t = e_t.permute(0, 2, 1)
      logits.append(logit)
      atts.append(e_t)
    logits = torch.cat(logits, dim = 1)
    atts = torch.cat(atts, dim = 1)

    return logits, atts

  def lstm(self, pre_cell, pre_hidden, pre_w, attended_feature): 
    
    # input_gate
    i_linearH = self.Hi_linear(pre_hidden)
    i_linearW = self.Wi_linear(pre_w)
    i_linearA = self.Ai_linear(attended_feature)
    input_gate = self.sigmoid(i_linearH + i_linearW + i_linearA) 

    # forget_gate
    f_linearH = self.Hf_linear(pre_hidden)
    f_linearW = self.Wf_linear(pre_w)
    f_linearA = self.Af_linear(attended_feature)
    forget_gate = self.sigmoid(f_linearH + f_linearW + f_linearA)

    # modulation_gate
    m_linearH = self.Hm_linear(pre_hidden)
    m_linearW = self.Wm_linear(pre_w)
    m_linearA = self.Am_linear(attended_feature)
    modulation_gate = self.tanh(m_linearH + m_linearW + m_linearA)

    # output_gate
    o_linearH = self.Ho_linear(pre_hidden)
    o_linearW = self.Wo_linear(pre_w)
    o_linearA = self.Ao_linear(attended_feature)
    output_gate = self.sigmoid(o_linearH + o_linearW + o_linearA)
    
    # hidden & cell
    fc_mul = torch.mul(forget_gate, pre_cell) 
    ig_mul = torch.mul(input_gate, modulation_gate) 
    cell = fc_mul + ig_mul
    hidden = torch.mul(output_gate, self.tanh(cell))

    return cell, hidden

  def soft_attention(self, feature, hidden):

    # attr_base
    hidden_base = np.zeros([self.batch_size, self.feature_num, self.lstm_hidden])
    hidden_base = torch.Tensor(hidden_base)
    hidden_base = hidden_base.to(device) #(2, 49, 1024)

    # base_clone
    hidden = hidden.unsqueeze(1)
    hidden_base[:, :, :] = hidden
    hidden_clone = hidden_base.clone()

    # e_t
    tanh_output = self.tanh(self.Wa_linear(feature) + self.Wh_linear(hidden_clone)) 
    e_t = self.We_linear(tanh_output)
    e_t = self.softmax(e_t)
    
    # att_output
    att_output = torch.mul(e_t, feature)
    att_output = att_output.sum(1)

    return att_output, e_t

  def caption_sampler(self, feature, input, length):

    # feature
    feature = feature.view(-1, self.feature_num, self.feature_dim) #torch.Size([2, 49, 2048])

    # start_base
    # start_word
    # start_wte
    start_base = torch.zeros([self.batch_size])
    start_word = torch.Tensor([self.start_word]) 
    start_base[:] = start_word
    start_base = start_base.long() 
    start_base = start_base.to(device) 
    start_wte = self.gpt2_wte(start_base) 

    # initial_state
    feature_mean = torch.mean(feature, dim = 1) 
    cell = self.cell_linear(feature_mean) 
    hidden = self.hidden_linear(feature_mean) #torch.Size([2, 1024])
    att_feature, e_t = self.soft_attention(feature, hidden)  #torch.Size([2, 2048]) torch.Size([2, 49, 1])

    # input
    # lstm_cell
    # get_next_index
    if input != None: # torch.Size([2, 3])
      sample_ids = []
      for k in range(input.shape[1]):
        orig_input = input[:, k].tolist() ##tensor([ 1, 11], device='cuda:0')  torch.Size([2])
        orig_input = torch.Tensor(orig_input)
        orig_input = orig_input.long()
        orig_input = orig_input.to(device)
        sample_ids.append(orig_input)
      
      input_emb = self.gpt2_wte(input) #torch.Size([2, 3, 768])
      input_wte = input_emb[:, -1, :] #torch.Size([2, 768])
      for i in range(length - input_emb.shape[1]):

        # lstm_cell
        cell, hidden = self.lstm(cell, hidden, input_wte, att_feature) #(2, 1024), (2, 1024)
        att_feature, e_t = self.soft_attention(feature, hidden) #torch.Size([2, 2048]) torch.Size([2, 49, 1])
        aW = self.Pa_linear(att_feature) 
        hW = self.Ph_linear(hidden) 
        logit = self.softmax(aW + hW) #torch.Size([2, 50260])

        # get_next_index
        next_index = self.get_next_index(logit) 
        next_index = torch.Tensor(next_index)
        next_index = next_index.to(device)
        next_index = next_index.long()
        sample_ids.append(next_index)
        input_wte = self.gpt2_wte(next_index)
        input_wte = input_wte.to(device)
      sample_ids = torch.stack(sample_ids, dim = 1)

    else:

      input_wte = start_wte #torch.Size([2, 768])
      sample_ids = []
      for j in range(length):

        # lstm_cell
        cell, hidden = self.lstm(cell, hidden, input_wte, att_feature) 
        att_feature, e_t = self.soft_attention(feature, hidden) #torch.Size([2, 49, 2048]) torch.Size([2, 1024])
        aW = self.Pa_linear(att_feature) 
        hW = self.Ph_linear(hidden) 
        logit = self.softmax(aW + hW) 

        # get_next_index
        next_index = self.get_next_index(logit) 
        next_index = torch.Tensor(next_index)
        next_index = next_index.to(device)
        next_index = next_index.long()
        sample_ids.append(next_index)
        input_wte = self.gpt2_wte(next_index)
        input_wte = input_wte.to(device)
      sample_ids = torch.stack(sample_ids, dim = 1)
 
    return sample_ids
      
  def get_next_index(self, input): #torch.Size([2, 50260])

    # sort input
    # zip value & index
    next_indexes = []
    for k in range(self.batch_size):
      list_zip = []
      input_list = input.tolist()
      input_sorted = sorted(input_list[k])
      for i in range(1, (self.index_num + 1)):
        value = input_sorted[-i]
        index = input_list[k].index(value)
        list_zip.append([value, index])

      # filter_index by value_min
      list_filtered = []
      for j in range(len(list_zip)):
        if list_zip[j][0] >= self.value_min:
          list_filtered.append(list_zip[j][1])
      if len(list_filtered) == 0:
        list_filtered = [list_zip[0][1]]
    
      # choose randomly
      next_index = random.choice(list_filtered)
      next_indexes.append(next_index)

    return next_indexes   
    
ATTEND_GAN_Generator = Caption_Generator().to(device)
