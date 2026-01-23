import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricContrastiveLoss(nn.Module):
    """
    Contrastive loss that:
    - Aligns responders (label=1) with their augmentations
    - Pushes responders away from non-responders (label=0)
    - Ignores non-responder ↔ non-responder relations
    """

    def __init__(self, margin=0.2, lambda_neg=1.0, timepoints=None):
        super().__init__()
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.timepoints = timepoints

    def forward(self, z, labels):
        """
        Args:
            z:        Tensor [B, D] - original embeddings
            z_aug:    Tensor [B, D] - augmented embeddings
            labels:   Tensor [B]    - binary labels (1=CR, 0=NR)
        """

        labels = labels.bool()
        timepoint_dim = z.shape[1] // self.timepoints

        # Normalize embeddings
        # z = F.normalize(z, dim=1)
        # z_aug = F.normalize(z_aug, dim=1)

        # ----------------------------------
        # Positive loss (responders only)
        # ----------------------------------
        if labels.any():
            # pos_sim = F.cosine_similarity(z[labels], z_aug[labels], dim=1) # only within a single responder patient
            # pos_sim = torch.matmul(z[labels], z_aug[labels].T)
            pos_sim = z[labels] @ z[labels].T
            pos_sim = pos_sim ** 2
            pos_sim = 1.0 - pos_sim

            B = pos_sim.size(0)
            mask = torch.eye(B, dtype=torch.bool, device=pos_sim.device)
            pos_sim = pos_sim.masked_fill(mask, float('-inf'))

            pos_sim = pos_sim.view(-1)
            pos_sim = pos_sim[~torch.isinf(pos_sim)]
            loss_align_positive = pos_sim.mean()
            
            ortho_loss_list = []
            for i in range(z[labels].shape[0]):
                # z_0 = z[labels][i][:timepoint_dim]
                # z_0 = F.normalize(z_0, dim=0)
                # z_1 = z[labels][i][timepoint_dim:2*timepoint_dim]
                # z_1 = F.normalize(z_1, dim=0)
                # z_2 = z[labels][i][2*timepoint_dim:]
                # z_2 = F.normalize(z_2, dim=0)

                # d1 = z_1 - z_0
                # d2 = z_2 - z_0
                # d1 = F.normalize(d1, dim=0)
                # d2 = F.normalize(d2, dim=0)
                # loss_ortho = (d1 @ d2) ** 2

                z_first = z[labels][i][:timepoint_dim]
                z_last = z[labels][i][-timepoint_dim:]
                loss_ortho = (F.cosine_similarity(z_first, z_last, dim=0)) ** 2
                # loss_ortho = torch.abs(F.cosine_similarity(z_first, z_last, dim=0))

                ortho_loss_list.append(loss_ortho)
            loss_orthogonal = torch.stack(ortho_loss_list).mean()
            
            
            temporal_loss_list = []
            for i in range(z[labels].shape[0]):
                embeddings = []
                for t in range(self.timepoints):
                    emb_t = z[labels][i][t * timepoint_dim : (t + 1) * timepoint_dim]
                    embeddings.append(emb_t)
                
                z_first_last = embeddings[-1] - embeddings[0]
                # z_first_last = F.normalize(z_first_last, dim=0)

                embeddings_diff = []
                for t in range(0, self.timepoints - 1):
                    diff = embeddings[t + 1] - embeddings[t]
                    # diff = F.normalize(diff, dim=0)
                    embeddings_diff.append(diff)
                
                embeddings_diff = sum(embeddings_diff)

                loss_temporal = (F.cosine_similarity(z_first_last, embeddings_diff, dim=0)) ** 2
                # loss_temporal = torch.abs(F.cosine_similarity(z_first_last, embeddings_diff, dim=0))
                temporal_loss_list.append(loss_temporal)
            loss_temporal = torch.stack(temporal_loss_list).mean()
            
            # loss_pos = loss_pos + ortho_loss + loss_temporal
            

            # loss_pos = (1.0 - pos_sim).mean()
        else:
            # loss_pos = torch.tensor(0.0, device=device)
            # loss_pos = z.sum() * 0.0
            # raise ValueError("No positive samples in the batch.")
            loss_align_positive = z.sum() * 0.0
            loss_orthogonal = z.sum() * 0.0
            loss_temporal = z.sum() * 0.0

        # ----------------------------------
        # Negative loss (CR vs NR)
        # ----------------------------------
        if labels.any() and (~labels).any():
            z_pos = z[labels]         # responders
            z_neg = z[~labels]       # non-responders

            neg_sim = z_pos @ z_neg.T
            neg_sim = neg_sim ** 2
            loss_align_neg = F.relu(neg_sim + self.margin).mean()
        else:
            # loss_neg = torch.tensor(0.0, device=device)
            # loss_neg = z.sum() * 0.0
            # raise ValueError("No negative samples in the batch.")
            loss_align_neg = z.sum() * 0.0

        # return loss_pos + self.lambda_neg * loss_neg
        return {
            'align': loss_align_positive + self.lambda_neg * loss_align_neg,
            'orthogonal': loss_orthogonal,
            'temporal': loss_temporal
        }

class CRSupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss applied ONLY to CR samples (label=1).
    Non-responders are completely ignored.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, labels):
        """
        Args:
            z:      Tensor [B, D] - embeddings (original OR augmented)
            labels: Tensor [B]    - binary labels (1=CR, 0=NR)
        """

        device = z.device
        labels = labels.bool()

        # Select CR samples only
        z_cr = z[labels]

        # Need at least 2 CR samples
        if z_cr.size(0) < 2:
            return z_cr.sum() * 0.0

        # z_cr = F.normalize(z_cr, dim=1)

        # Cosine similarity matrix
        sim = torch.matmul(z_cr, z_cr.T) / self.temperature

        # Mask self-similarity
        mask = torch.eye(sim.size(0), device=device).bool()
        sim.masked_fill_(mask, -1e9)

        # Log-softmax
        log_prob = F.log_softmax(sim, dim=1)

        # All other CR samples are positives
        loss = -log_prob.mean()

        return loss
