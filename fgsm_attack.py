# FGSM attack code
import torch

def fgsm_step(image, epsilon, data_grad):
    # element-wise sign of the data gradient
    sign_data_grad = torch.sign(data_grad);
    
    # perturbing image
    perturbed_image = image + epsilon*sign_data_grad;
    
    # clamp to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -0.6, 0.6);
    
    return perturbed_image

def test(model, device, test_loader, epsilon, cuda_avail):
    # accuracy counter
    correct = 0;
    adv_examples = [];

    # loop over all examples in test set
    for data, target in test_loader:
        
        # send the data and label to the device
        if cuda_avail:
            data, target = data.cuda(), target.cuda();
        
        # normalizing image to [-0.6, 0.6], since that bound were used in training
        data -= 0.5;
        data /= 0.5;
        data *= 0.6;
        
        # allowing to compute gradients through input tensor
        data.requires_grad = True;

        # forward pass
        output = model(data);
        init_pred = output.max(1, keepdim=True)[1]; # get the index of the max logit

        # if the initial prediction is wrong, there is no need to perform the attack
        if init_pred.item() != target.item():
            continue

        # calculate the loss
        loss = torch.nn.functional.nll_loss(torch.softmax(output, -1), target);

        # zero all existing gradients
        model.zero_grad();

        # calculate gradients of model in backward pass
        loss.backward();

        # collect datagrad
        data_grad = data.grad.data

        # call FGSM Attack (with scaled epsilon, since the input data is also scaled)
        perturbed_data = fgsm_step(data, epsilon*1.2, data_grad)
        
        # reclassify the perturbed image
        output = torch.sigmoid(model(perturbed_data))

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy();
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) );
        else:
            # save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader));
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc));

    # return the accuracy and an adversarial example
    return final_acc, adv_examples;