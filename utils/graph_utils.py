import torch
from torch_geometric.data import Data, Batch
import torch.nn as nn

def make_directed_complete_forward_graph(embeddings, batch_size=None):
    """
    embeddings: [N, D]
    batch_size: int or None
    returns a directed complete graph (acyclic): edges from i -> j only if i < j
    """

    if batch_size is not None:
        embeddings = embeddings.view(batch_size, -1, embeddings.size(-1))

    data_list = []
    for i in range(batch_size):

        x = embeddings[i]
        N = x.size(0)

        src_list = []
        dst_list = []

        for i in range(N):
            for j in range(i + 1, N):
                src_list.append(i)
                dst_list.append(j)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    return Batch.from_data_list(data_list)


def compute_graph_alignment_loss(
    latents: torch.Tensor,
    labels: torch.Tensor,
    align_labels: list,
) -> torch.Tensor:
    """
    Compute alignment loss on a population level.

    The alignment loss is computed as the average cosine distance
    (1 - cosine similarity) between latent representations that share
    the same label.

    Args:
        latents (torch.Tensor): Latent representations of shape (N, D).
        labels (torch.Tensor): Labels of shape (N,).
        align_labels (list): Labels to compute alignment loss for.

    Returns:
        torch.Tensor: Scalar alignment loss with valid grad_fn.
    """

    # Graph-connected zero (IMPORTANT)
    loss_align_total = latents.sum() * 0.0
    num_effective_labels = 0

    for align_label in align_labels:
        indices = (labels == align_label).nonzero(as_tuple=True)[0]
        latents_label = latents[indices]

        if latents_label.size(0) < 2:
            continue

        cosine_distances = []

        for i in range(latents_label.size(0)):
            for j in range(i + 1, latents_label.size(0)):
                cosine_distance = 1.0 - nn.functional.cosine_similarity(
                    latents_label[i].unsqueeze(0),
                    latents_label[j].unsqueeze(0),
                    dim=-1,
                )
                cosine_distances.append(cosine_distance)

        if len(cosine_distances) > 0:
            loss_label = torch.stack(cosine_distances).mean()
            loss_align_total = loss_align_total + loss_label
            num_effective_labels += 1

    if num_effective_labels > 0:
        loss_align_total = loss_align_total / num_effective_labels

    return loss_align_total


        