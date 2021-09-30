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
