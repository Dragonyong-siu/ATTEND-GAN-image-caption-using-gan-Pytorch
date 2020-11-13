#7) ATTEND_GAN_Loss
 #7.1) generator_loss : L2
  #7.1.1) crossentropy function 
  #7.1.2) (1 - x)^2 function

 #7.2) generator_loss : L1
  #7.2.1) connect with discriminator : reward

 #7.3) discriminator_loss : binary crossentropy function

 #7.4) get_reward : calculate policy gradient to get reward
  #7.4.1) iterate MC_num times to decrease the variance of the next words

class loss_set():
  def generator_L2(self, logit, att, target, lambda1, ):

    #7.1) generator_loss : L2
     #7.1.1) crossentropy function
    loss_function = nn.CrossEntropyLoss()
    p_loss = loss_function(logit, target)

    #7.1) generator_loss : L2
     #7.1.2) (1 - x)^2 function
    att_per_word = att.sum(1) #torch.Size([2, 49])
    one = torch.Tensor([16. / 49])
    one = one.to(device)

    # multiply lambda to balance and then get L2 by sum
    sub = one - att_per_word
    a_loss = torch.mul(sub, sub) #torch.Size([2, 49])
    a_loss = a_loss.sum(1) #torch.Size([2])
    L2_loss = p_loss + torch.mean(torch.mul(lambda1, a_loss))

    return L2_loss

  def generator_L1(self, logit, action, reward): #reward - torch.Size([2, 16])
    
    #7.2) generator_loss : L1
     #7.2.1) connect with discriminator : reward
    loss_function = nn.CrossEntropyLoss(reduction = 'none')
    loss = loss_function(logit, action)
    L1_loss = torch.matmul(loss, reward)
    L1_loss = torch.mean(L1_loss)

    return L1_loss
  
  def discriminator_L(self, pred, target):

    #7.3) discriminator_loss : binary crossentropy function
    loss_function = nn.BCELoss()
    d_loss = loss_function(pred, target)

    return d_loss
  
  def get_reward(self, input, MC_num, feature, generator, discriminator):

    #7.4) get_reward : calculate policy gradient to get reward
    batch_size, seq_len = input.shape[0], input.shape[1]
    rewards = []
    
    #7.4.1) iterate MC_num times to decrease the variance of the next words
    for i in range(MC_num):
      for j in range(1, seq_len):
        seq_part = input[:, :j]
        sample = generator.caption_sampler(feature, seq_part, seq_len)
        reward = discriminator(sample) #(2, 1)
        reward = reward.detach().cpu().numpy() #(2, 1)
        if i == 0:
          rewards.append(reward)
        else:
          rewards[j - 1] += reward
      
      reward = discriminator(input)
      reward = reward.squeeze(0)
      reward = reward.detach().cpu().numpy() 
      if i == 0:
        rewards.append(reward)
      else:
        rewards[seq_len - 1] += reward

    rewards = np.concatenate(rewards, axis = 1)
    rewards = torch.Tensor(rewards)
    rewards = torch.divide(rewards, 1.0 * MC_num)

    return rewards

ATTEND_GAN_Loss = loss_set()
