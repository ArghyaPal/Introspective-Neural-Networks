import torch

def NormalizeData(data, to_min=-0.6, to_max=0.6):
    for j in range(data.shape[0]):
        a = data[j, ...].min();
        b = data[j, ...].max();
        data[j, ...] -= a;
        data[j, ...] /= (b-a);
        data[j, ...] *= (to_max-to_min);
        data[j, ...] -= to_max;
    return data;

def Sample(module, num_iter=10, learning_rate=0.01, add_noise=True, early_stopping=True, init_std=0.3):
    assert isinstance(module.X, torch.nn.Parameter), 'Expected X to be an instance of torch.nn.Parameter';

    module.train(False);

    # we do not want to create a graph and do backprop on net parameters, since we need only gradient of X
    for name, param in module.named_parameters():
        if name != 'X':
            param.requires_grad = False;
        else:
            param.requires_grad = True;
    
    # initialize from Gaussian distribution
    module.X.data = module.X.data.normal_(mean=0, std=init_std);
    
    # initialize optimizer to minimize the objective function
    opt = torch.optim.Adam([module.X], lr=learning_rate, amsgrad=True, betas=(0.5, 0.999));
    std_noise = learning_rate;
    
    for i in range(num_iter):
        # zeroing gradients of parameter X
        opt.zero_grad();
        
        # perform forward pass
        classes = -torch.sum(torch.diagonal(module.to_synth()));
        
        # perform backward pass
        classes.backward();
           
        # Langevin dynamics, if needed - scaling the step size and adding Gaussian noise to gradients
        if add_noise:
            module.X.grad += torch.empty_like(module.X.data).normal_(mean=0, std=0.1*opt.param_groups[0]['lr']);
            opt.param_groups[0]['lr'] *= 0.92;
        
        # do gradient step
        opt.step()
        
        # normalizing data to stay in needed domain
        module.X.data = NormalizeData(module.X.data);

        # early stopping if needed
        if early_stopping and torch.all(torch.argmax(torch.sigmoid(module(module.X.detach().data)).data, dim=1) == torch.arange(0,10).cuda()):
            break
    # preparing to the next training iteration
    module.train(True);

    # we finished the sampling procedure and we now do want to create a graph and do backprop on net parameters, since we need to train it further
    for name, param in module.named_parameters():
        if name != 'X':
            param.requires_grad = True;
        else:
            param.requires_grad = False;

    return module.X.data;