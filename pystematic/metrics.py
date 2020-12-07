import torch


def accuracy(output, target, all_reduce=True):
    """Calculates the accuracy
    :param output: output from model
    :param target: torch.LongTensor
    """
    return top_k_accuracy(output, target, 1, all_reduce)

def accuracy_partial(output, target, all_reduce=True):
    """Calculates the (partial) accuracy
    :param output: output from model
    :param target: torch.LongTensor
    """
    return top_k_accuracy_partial(output, target, 1, all_reduce)


def top_k_accuracy(output, target, k, all_reduce=True):
    num_correct, num_examples = top_k_accuracy_partial(output, target, k, all_reduce)

    return num_correct / num_examples


def top_k_accuracy_partial(output, target, k, all_reduce=True):
    
    top_k_values, top_k_classes = torch.topk(output, k, dim=1)

    num_correct = torch.sum(torch.eq(top_k_classes, target.unsqueeze(1))).float()
    num_examples = torch.tensor(torch.numel(target), device=output.device).float()

    if torch.distributed.is_initialized() and all_reduce:
        torch.distributed.all_reduce(num_correct, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_examples, op=torch.distributed.ReduceOp.SUM)

    return num_correct, num_examples