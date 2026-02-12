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

    def __init__(self, margin=0.2, lambda_neg=1.0, timepoints=None, skip_loss=None, temperature=1.0, feature_sim='cosine'):
        super().__init__()
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.timepoints = timepoints
        self.skip_loss = skip_loss
        self.temperature = temperature
        self.feature_sim = feature_sim
        
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
            # pos_sim = z[labels] @ z[labels].T
            # pos_sim = pos_sim ** 2
            # pos_sim = 1.0 - pos_sim

            z_positive = z[labels]
            perm_latents_1 = torch.randperm(z_positive.size(0))
            z_positive_permuted = z_positive[perm_latents_1]
            loss_align_positive = 1.0 - nn.functional.cosine_similarity(z_positive, z_positive_permuted, dim=-1).mean()

            if self.skip_loss == "alignment_loss":
                loss_align_positive = z.sum() * 0.0

            # B = pos_sim.size(0)
            # mask = torch.eye(B, dtype=torch.bool, device=pos_sim.device)
            # pos_sim = pos_sim.masked_fill(mask, float('-inf'))

            # pos_sim = pos_sim.view(-1)
            # pos_sim = pos_sim[~torch.isinf(pos_sim)]
            # loss_align_positive = pos_sim.mean()
            
            ortho_loss_list = []
            for i in range(z[labels].shape[0]):
                z_0 = z[labels][i][:timepoint_dim]
                z_1 = z[labels][i][timepoint_dim:2*timepoint_dim]
                z_2 = z[labels][i][2*timepoint_dim:3*timepoint_dim]
                z_3 = z[labels][i][-timepoint_dim:]

                l1 = torch.abs(F.cosine_similarity(z_0, z_1, dim=0))
                l2 = torch.abs(F.cosine_similarity(z_1, z_2, dim=0))
                l3 = torch.abs(F.cosine_similarity(z_2, z_3, dim=0))
                # loss_ortho = (l1 + l2 + l3 ) / 3.0

                l4 = torch.abs(F.cosine_similarity(z_0, z_2, dim=0))
                l5 = torch.abs(F.cosine_similarity(z_0, z_3, dim=0))
                l6 = torch.abs(F.cosine_similarity(z_1, z_3, dim=0))                
                loss_ortho = (l1 + l2 + l3 + l4 + l5 + l6) / 6.0

                ortho_loss_list.append(loss_ortho)

            loss_orthogonal = torch.stack(ortho_loss_list).mean()            
            
            if self.skip_loss == "orthogonality_loss":
                loss_orthogonal = z.sum() * 0.0
            
            ####################################################
            # Uncomment the block below for original version
            ###################################################

            # temporal_loss_list = []
            # for i in range(z[labels].shape[0]):
            #     z_0 = z[labels][i][:timepoint_dim]
            #     z_1 = z[labels][i][timepoint_dim:2*timepoint_dim]
            #     z_2 = z[labels][i][2*timepoint_dim:3*timepoint_dim]
            #     z_3 = z[labels][i][-timepoint_dim:]

            #     z_10 = z_1 - z_0
            #     z_20 = z_2 - z_0
            #     z_30 = z_3 - z_0

            #     z_21 = z_2 - z_1
            #     z_31 = z_3 - z_1
            #     z_32 = z_3 - z_2

            #     l1 = 1.0 - F.cosine_similarity(z_10 + z_21 + z_32, z_30, dim=0)
            #     l2 = 1.0 - F.cosine_similarity(z_10 + z_21, z_20, dim=0)
            #     l3 = 1.0 - F.cosine_similarity(z_21 + z_32, z_31, dim=0)

            #     loss_temporal = (l1 + l2 + l3) / 3.0
            #     temporal_loss_list.append(loss_temporal)                
                
            # loss_temporal = torch.stack(temporal_loss_list).mean()  
            
            # modified original version
            temporal_loss_list = []
            for i in range(z[labels].shape[0]):
                z_0 = z[labels][i][:timepoint_dim]
                z_1 = z[labels][i][timepoint_dim:2*timepoint_dim]
                z_2 = z[labels][i][2*timepoint_dim:3*timepoint_dim]
                z_3 = z[labels][i][-timepoint_dim:]

                z_10 = z_1 - z_0
                z_21 = z_2 - z_1
                z_32 = z_3 - z_2

                loss_temporal = 1.0 - F.cosine_similarity(z_10 + z_21 + z_32, z_3, dim=0)

                temporal_loss_list.append(loss_temporal)                
                
            loss_temporal = torch.stack(temporal_loss_list).mean()  

            ######################################
            # rank-n-based temporal loss
            #######################################
            # temporal_loss_fn = RnCLoss(temperature=self.temperature, label_diff='l1', feature_sim='cosine') 

            # temporal_loss_list = []
            # for i in range(z[labels].shape[0]):
            #     z_0 = z[labels][i][:timepoint_dim]
            #     z_1 = z[labels][i][timepoint_dim:2*timepoint_dim]
            #     z_2 = z[labels][i][2*timepoint_dim:3*timepoint_dim]
            #     z_3 = z[labels][i][-timepoint_dim:]

            #     features_temp = torch.stack([z_0, z_1, z_2, z_3], dim=0)  # [4, feat_dim]
            #     labels_temp = torch.tensor([0, 1, 2, 3], device=z.device)  # [4]
            #     labels_temp = labels_temp.unsqueeze(-1)  # [4, 1]

            #     loss_temp = temporal_loss_fn(features_temp, labels_temp)
            #     temporal_loss_list.append(loss_temp)

            # loss_temporal = torch.stack(temporal_loss_list).mean()
        

            if self.skip_loss == "temporal_loss":
                loss_temporal = z.sum() * 0.0         
            
        else:            
            loss_align_positive = z.sum() * 0.0
            loss_orthogonal = z.sum() * 0.0
            loss_temporal = z.sum() * 0.0

        # ----------------------------------
        # Negative loss (CR vs NR)
        # ----------------------------------
        if labels.any() and (~labels).any():
            z_pos = z[labels]         # responders
            z_neg = z[~labels]       # non-responders

            # neg_sim = z_pos @ z_neg.T
            # loss_align_negative = torch.abs(neg_sim).mean()
            # loss_align_negative = F.relu(neg_sim + self.margin).mean()

            n_pos = z_pos.size(0)
            n_neg = z_neg.size(0)

            if n_neg < n_pos:
                permutation = torch.randperm(z_pos.size(0))
                z_pos = z_pos[permutation]
                z_pos = z_pos[:z_neg.shape[0], :]
            elif n_pos < n_neg:
                permutation = torch.randperm(z_neg.size(0))
                z_neg = z_neg[permutation]
                z_neg = z_neg[:z_pos.shape[0], :]
            else:
                pass # equal sizes            

            loss_align_negative = nn.functional.cosine_similarity(z_pos, z_neg, dim=-1).mean()

            if self.skip_loss == "alignment_loss":
                loss_align_negative = z.sum() * 0.0

        else:            
            loss_align_negative = z.sum() * 0.0

        return {
            'align': loss_align_positive + loss_align_negative,
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


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)

class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'cosine':
            return F.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='cosine'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        # features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        # labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
