import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from fs_ot import *
from evaluation_metrics import *
from training_config import *


def FGSM(net,inputs,labels,input_lengths, label_lengths,device,eps,criterion):
    
    net.train()
    inputs.requires_grad = True
    
    max_v,min_v = inputs.max().detach().cpu().item(),inputs.min().detach().cpu().item()
    
    alpha = eps 
    output = net(inputs)  # (batch, time, n_class)
    output = F.log_softmax(output, dim=2)
    output = output.transpose(0, 1) # (time, batch, n_class)

    loss = criterion(output, labels, input_lengths, label_lengths)

    grad = torch.autograd.grad(loss, inputs,
                                   retain_graph=False, create_graph=False)[0]

    x = inputs + alpha * grad.sign()
    x = torch.clamp(x,min=min_v,max=(max_v/2)).detach()
    
    return x



def PGD(net,inputs,labels,input_lengths, label_lengths,device,eps,iters,alpha,criterion):
        
        net.train()
        
        inputs,labels = inputs.to(device),labels.to(device)
        ori_images = inputs.clone().detach()
        max_v,min_v = inputs.max().detach().cpu().item(),inputs.min().detach().cpu().item()    
            
        inputs = inputs + torch.empty_like(inputs).uniform_(-eps, eps)
        inputs = torch.clamp(inputs, min=min_v, max=(max_v))
        
        for i in range(iters) :    
            inputs.requires_grad = True
            
            output = net(inputs)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            cost = criterion(output, labels, input_lengths, label_lengths).to(device)
            
            #cost = criterion(outputs, labels).to(device)
            
            grad = torch.autograd.grad(cost, inputs, 
                                       retain_graph=False, create_graph=False)[0]

            adv_images = inputs + alpha*grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            inputs = torch.clamp(ori_images + eta, min=min_v, max=max_v).detach()

        adv_images = inputs
        
        return adv_images  


    
def mixPGD(net,inputs,labels,input_lengths, label_lengths,device,eps,iters,alpha,criterion):
        
        net.train()
        inputs,labels = inputs.to(device),labels.to(device)
        batch_size = inputs.shape[0]
        ori_images = inputs.clone().detach()
        output_nat = net(inputs)        
            
        max_v,min_v = inputs.max().detach().cpu().item(),inputs.min().detach().cpu().item()    
        
        x_adv = inputs.detach() + 0.001 * torch.randn(inputs.shape).to(device).detach()
        
        for i in range(iters) :    
            x_adv.requires_grad = True
            
            output = net(x_adv)  # (batch, time, n_class)
            
            ot_cost = sinkhorn_loss_joint_IPOT(1, 0.00, output_nat,
                                                  output, None, None,
                                                  0.01, batch_size, batch_size, device)
            
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            cost = criterion(output, labels, input_lengths, label_lengths).to(device)
            
                        
            total_cost = ot_cost + cost
            
            grad = torch.autograd.grad(total_cost, x_adv, 
                                       retain_graph=False, create_graph=False)[0]

            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, inputs - eps), inputs + eps)
            x_adv = torch.clamp(x_adv, min_v, max_v)
            
        return x_adv 
    

def FeatureScatter(net,inputs,labels,device,input_lengths,label_lengths,eps,iters,alpha,criterion,text_transform):
    
    inputs,labels = inputs.to(device),labels.to(device)
    max_v,min_v = inputs.max().detach().cpu().item(),inputs.min().detach().cpu().item()     
    
    logits = net(inputs)
    num_classes = logits.size(2)
    outputs = net(inputs)
    
    x = inputs.detach()
    x_org = x.detach()
    
    x = x + torch.zeros_like(x).uniform_(-eps,eps)

    logits_pred_nat = net(inputs)
        
    logits_pred_nat = logits_pred_nat.squeeze()
    batch_size = logits_pred_nat.size(0)
    
    m,n = batch_size,batch_size
    
    iter_num = iters 
        
    for i in range(iter_num):
            x.requires_grad_()
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred = net(x)

            logits_pred = logits_pred.squeeze()

            ot_loss = sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n, device)
            

            net.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward(retain_graph=True)
            
            x_adv = x.data + alpha * torch.sign(x.grad.data)  
            x_adv = torch.min(torch.max(x_adv, inputs - eps),
                              inputs + eps)
            x_adv = torch.clamp(x_adv,min=min_v,max=max_v/2)  
            x = Variable(x_adv)

            logits_pred = net(x)
            logits_pred = logits_pred.transpose(0,1)
            
            net.zero_grad()

            
            adv_loss = criterion(logits_pred, labels, input_lengths, label_lengths).to(device) 
            
    return x_adv, adv_loss





















