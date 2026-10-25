from typing import Tuple,Dict,List
from torch.nn import Module
import torch

class MultiLossDynamicBalancing(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,individual_losses : List[torch.Tensor]) -> torch.Tensor:
        task_difficulties = [loss.item() for loss in individual_losses]
        task_weights = 1.0 / (torch.tensor(task_difficulties) + 1e-6)  # Normalize inverse difficulties
        combined_loss = sum(task_weights[i] * individual_losses[i] for i in range(len(individual_losses)))
        return combined_loss

class MultiLossWeightedSum(Module):
    def __init__(self,num_properties:int,device = torch.device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = torch.ones(num_properties).to(device)

    def forward(self,individual_losses : List[torch.Tensor]) -> torch.Tensor:
        combined_loss = sum(self.loss_weights[i] * individual_losses[i] for i in range(len(individual_losses)))
        return combined_loss

class MultiLossUncertaintyBased(Module):
    def __init__(self, num_properties:int,device = torch.device,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.uncertainties = torch.nn.Parameter(torch.zeros(num_properties), requires_grad=True).to(
                device)

    def forward(self,individual_losses : List[torch.Tensor]) -> torch.Tensor:
        # Uncertainty-based weighting (as per Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses")
        combined_loss = sum(
            (1 / (2 * torch.exp(self.uncertainties[i]))) * individual_losses[i] + self.uncertainties[i]
            for i in range(len(individual_losses))
        )
        return combined_loss

class MultiLossSimpleSum(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,individual_losses : List[torch.Tensor]) -> torch.Tensor:
        combined_loss = sum(individual_losses)
        return combined_loss


if __name__ == "__main__":
    tensor_list = [torch.tensor(5),torch.tensor(3),torch.tensor(10)]
    loss_aggre = MultiLossDynamicBalancing()
    loss = loss_aggre(tensor_list)
    print(loss)
