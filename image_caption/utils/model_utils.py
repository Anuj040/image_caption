"""attention map with alibi biases
https://github.com/ofirpress/attention_with_linear_biases/blob/1c8ceb539458204d5c875d74abfddcb5b64ae8c9/fairseq/models/transformer.py#L742
https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L1011
"""
import math

import torch


def get_slopes(num_heads: int) -> list:
    """
    Calculates head dependent linear bias scalar for multihead attention as described in
    Train Short, Test Long: Attention with Linear Biases (ALiBi) Enables Input Length Extrapolation
    (https://ofir.io/train_short_test_long.pdf)

    Args:
        num_heads (int): number of separate attention heads in the multihead attention layer
    """

    def get_slopes_power_of_2(num_heads: int) -> list:
        """calculates scalars when number of heads is a power of 2"""
        start = 2 ** (-(2 ** -(math.log2(num_heads) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(num_heads)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    # The original paper only train models that have 2^a heads for some a. This function has
    # some good properties that only occur when the input is a power of 2. To maintain that even
    # when the number of heads is not a power of 2, we use this workaround.
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + get_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
    )


def fill_with_neg_inf(tensor: torch.Tensor) -> torch.Tensor:
    """FP16-compatible function that fills a tensor with -inf."""
    return tensor.float().fill_(float("-inf")).type_as(tensor)


def get_alibi_mask(inputs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """generates the alibi mask as described in
    Train Short, Test Long: Attention with Linear Biases (ALiBi) Enables Input Length Extrapolation
    (https://ofir.io/train_short_test_long.pdf)

    Args:
        inputs (torch.Tensor): embedding tensor (B * Ns * embd_dim)
        num_heads (int): #attention heads

    Returns:
        torch.Tensor: mask tensor for each atention head
    """

    batch_size, maxpos, _ = inputs.size()
    slopes = torch.Tensor(get_slopes(num_heads))
    # In the next line, part after * is constructs the diagonal matrix (right matrix in Fig 3).
    # It doesn't give the same matrix as in Fig 3, but one where all rows are identical.
    # It works because softmax is invariant to translation, and bias functions are always linear.
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(
        0
    ).unsqueeze(0).expand(num_heads, -1, -1)
    alibi = alibi.view(num_heads, 1, maxpos)
    alibi = alibi.repeat(batch_size, 1, 1)  # batch_size, 1, 1
    future_mask = torch.triu(fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
    future_mask = future_mask.unsqueeze(0) + alibi
    return future_mask


# pylint: disable = attribute-defined-outside-init
class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """resets the metric values"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """updates the metric values

        Args:
            val (float): latest metric value
            n (int, optional): metric counter. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(
    data_name,
    epoch,
    epochs_since_improvement,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    bleu4,
    is_best,
):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "bleu-4": bleu4,
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }
    filename = "checkpoint_" + data_name + ".pth.tar"
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, "BEST_" + filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    l_rate = optimizer.param_groups[0]["lr"]
    print(f"New learning rate: {l_rate}\n")
