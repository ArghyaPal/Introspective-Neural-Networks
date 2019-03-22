# adopted from https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
import torch
from tqdm.autonotebook import tqdm

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro");
    return metric_fn(true_y, pred_y);
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")
    pass
                           

def measure_scores(model, val_loader, cuda_avail):
    model.train(False);
    val_batches = len(val_loader);
    loss_function = torch.nn.CrossEntropyLoss();
    torch.cuda.empty_cache();

    val_losses = 0;
    precision, recall, f1, accuracy = [], [], [], [];

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if cuda_avail:
                X, y = data[0].cuda(), data[1].cuda().squeeze(-1).long();
            else:
                X, y = data[0], data[1].squeeze(-1).long();

            outputs = torch.sigmoid(model(X));
            val_losses += loss_function(outputs, y.long());

            predicted_classes = torch.max(torch.sigmoid(outputs), 1)[1];

            for acc, metric in zip((precision, recall, f1, accuracy), 
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(calculate_metric(metric, y.cpu(), predicted_classes.cpu()));
    print_scores(precision, recall, f1, accuracy, val_batches);
    model.train(True);
    return val_losses/val_batches, sum(accuracy)/val_batches;

def train(model, train_loader, val_loader, epochs=1, alpha=0.9, lrate=1e-3, stop_accuracy=0.9940, cuda_avail=True):
    start_ts = time.time();
    losses = [];
    nll_loss_function = torch.nn.NLLLoss();
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=lrate);

    batches = len(train_loader);
              
    # training loop + eval loop
    for epoch in range(epochs):
        total_loss = 0;
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches);
        model.train(True);

        for i, data in progress:
            
            if cuda_avail:
                X, y, fakes = data[0].cuda(), data[1].cuda().squeeze(-1).long(), data[2].cuda();
            else:
                X, y, fakes = data[0], data[1].squeeze(-1).long(), data[2];
            
            model.zero_grad();
            outputs = model(X);
            
            # prepare loss
            loss = 0;
            
            # computing loss on fake samples
            if not torch.all(fakes == 0):
                mask = torch.zeros((y.shape[0], 10)).byte();
                for i in range(y.shape[0]):
                    mask[i, torch.abs(y[i])] = 1;
                loss = alpha*torch.sum(torch.log1p(torch.exp(torch.sigmoid(outputs[mask][fakes]))));
            
            # computing loss on real samples  
            if not torch.all(fakes == 1):
                loss += (1-alpha)*nll_loss_function(torch.softmax(torch.sigmoid(outputs[1-fakes]), -1), y[1-fakes]);
            
            total_loss += loss.detach().data;
            loss.backward()
            optimizer.step()
            progress.set_description("Loss: {:.4f}".format(loss.item()))

        val_loss, val_accuracy = measure_scores(model, val_loader, cuda_avail);
        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_loss}")
        losses.append(total_loss/batches)
        
        # early stopping with a threshold
        if val_accuracy >= stop_accuracy:
            break
    pass