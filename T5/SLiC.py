import torch
import torch.nn as nn
import torch.nn.functional as F

# class SLiC(nn.Module):
#     """
#     base_model == calibration_model(having parameter of fine_tuned with MLE)
#     """
    
#     def __init__(self) -> None:
#         super().__init__()
        
#     def forward(self, base_model, ft_model, prompt, positive_candidate, negative_candidate, label=None, alpha=0.01):
#         calibration_loss = self._rank_loss(base_model,
#                                 prompt,
#                                 positive_candidate,
#                                 negative_candidate)
    
#         regularization_loss = self._KLD_loss(base_model, ft_model, prompt, label)
        
#         return calibration_loss + (alpha * regularization_loss)
        
#     def _compute_seq_log_prob(self, base_logits, target_ids):
#         log_prob = F.log_softmax(input=base_logits, dim=-1)
#         log_prob = log_prob.gather(dim=-1, index=target_ids)
#         seq_log_prob = log_prob.sum(dim=-1)
        
#         return seq_log_prob

#     def _rank_loss(self, base_model, prompt, positive_candidate, negative_candidate):
#         base_logits = base_model.forward(prompt).logits

#         log_prob_positive = self._compute_seq_log_prob(base_logits, positive_candidate)
#         log_prob_negative = self._compute_seq_log_prob(base_logits, negative_candidate)

#         loss = F.relu(-log_prob_positive + log_prob_negative)

#         return loss

#     def _KLD_loss(self, base_model, ft_model, prompt, label):
#         target = base_model.forward(prompt, label).logits
#         inputs = ft_model.forward(prompt, label).logits
        
#         KLD = nn.KLDivLoss(reduction="batchmean")
#         loss = KLD(inputs, target)
        
#         return loss

# class SLiC(nn.Module):
#     """
#     base_model == calibration_model(having parameter of fine_tuned with MLE)
#     """
    
#     def __init__(self) -> None:
#         super().__init__()
        
#     def forward(self, base_model, prompt, prompt_attn, positive_candidate, negative_candidate, labels, alpha=0.5):
#         positive_base_logits = base_model(prompt, prompt_attn, labels=positive_candidate).logits
#         negative_base_logits = base_model(prompt, prompt_attn, labels=negative_candidate).logits
#         base_loss = base_model(prompt, prompt_attn, labels=labels).loss
        # base_logits, base_loss = base_output.logits, base_output.loss
        # ft_logits = ft_model.model(prompt, prompt_attn, labels=labels).logits # ?
        
        # calibration_loss = self._rank_loss(base_logits,
        #                         positive_candidate,
        #                         negative_candidate)
        # calibration_loss = self._rank_loss(positive_base_logits,
        #                                    negative_base_logits,
        #                                     positive_candidate,
        #                                     negative_candidate)
    
        # regularization_loss = self._KLD_loss(base_logits, ft_logits, labels)
        # regularization_loss = base_loss
        
        # return calibration_loss + (alpha * regularization_loss)
        
    # def _compute_seq_log_prob(self, base_logits, target_ids):
    #     log_prob = F.log_softmax(input=base_logits, dim=-1)
        # max_len - decoder max len = 69
        # padded_log_prob = F.pad(log_prob, pad=(0,0,0,69), mode='constant', value=0)
        # padded_log_prob = padded_log_prob.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    #     log_prob = log_prob.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        
    #     seq_log_prob = torch.sum(log_prob, dim=-1)
        
    #     return seq_log_prob

    # def _rank_loss(self, positive_base_logits, negative_base_logits, positive_candidate, negative_candidate):

    #     log_prob_positive = self._compute_seq_log_prob(positive_base_logits, positive_candidate)
    #     log_prob_negative = self._compute_seq_log_prob(negative_base_logits, negative_candidate)

    #     loss = F.relu(-log_prob_positive + log_prob_negative)
    #     loss.mean()

    #     return loss

    # def _KLD_loss(self, base_logits, ft_logits, labels):
    #     target = F.softmax(base_logits, dim=-1)
    #     inputs = F.log_softmax(ft_logits, dim=-1)
        
    #     target = target.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    #     inputs = inputs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
    #     KLD = nn.KLDivLoss(reduction="batchmean")
    #     loss = KLD(inputs, target) # inputs = pred, target = true
        
    #     return loss

class SLiC(nn.Module):
    """
    base_model == calibration_model(having parameter of fine_tuned with MLE)
    """
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, base_model, prompt, prompt_attn, positive_candidate, negative_candidate, labels, alpha=0.5):
        positive_base_logits = base_model(prompt, prompt_attn, labels=positive_candidate).logits
        negative_base_logits = base_model(prompt, prompt_attn, labels=negative_candidate).logits
        base_loss = base_model(prompt, prompt_attn, labels=labels).loss
        
        calibration_loss = self._rank_loss(positive_base_logits,
                                           negative_base_logits)
        regularization_loss = base_loss
        
        return calibration_loss + (alpha * regularization_loss)
        
    def _compute_seq_log_prob(self, base_logits, target_ids):
        log_prob = F.log_softmax(input=base_logits, dim=-1)
        # max_len - decoder max len = 69
        padded_log_prob = F.pad(log_prob, pad=(0,0,0,69), mode='constant', value=0)
        padded_log_prob = padded_log_prob.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        
        seq_log_prob = padded_log_prob.sum(dim=-1)
        
        return seq_log_prob

    def _rank_loss(self, positive_candidate, negative_candidate):

        loss = F.relu(-positive_candidate + negative_candidate)
        loss = loss.mean()

        return loss