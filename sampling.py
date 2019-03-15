import torch
    
class Synthesis():
    def __init__(self, init_std=0.3, noise_grad_std=None):
        self.init_std = init_std;
        self.noise_grad_std = noise_grad_std;
        
    def sample(self, module, num_iter=10, learning_rate=1e-2):
        assert isinstance(module.X, torch.nn.Parameter), 'Expected X to be an instance of torch.nn.Parameter';
        
        # we do not want to create a graph and do backprop on net parameters, since we need only gradient of X
        for name, param in module.named_parameters():
            if name != 'X':
                param.requires_grad = False;
            else:
                param.requires_grad = True;
        
        module.X.data = module.X.data.normal_(mean=0, std=self.init_std);
        opt = torch.optim.SGD([module.X], lr=learning_rate);
        
        for i in range(num_iter):
            opt.zero_grad();
            classes = -module.to_synth();
            for j in range(classes.shape[0]):
                classes[j, ...].backward(retain_graph=True);
                
            if self.noise_grad_std is not None:
                module.X.data += torch.empty_like(module.X.data).normal_(mean=0, std=self.noise_grad_std);
            
            opt.step();
        
        # we want to perform the training of the net and do not need to backprop through X
        for name, param in module.named_parameters():
            if name != 'X':
                param.requires_grad = True;
            else:
                param.requires_grad = False;
                
        return module.X.data;
