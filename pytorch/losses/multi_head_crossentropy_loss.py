from typing import List, Optional
import torch.nn as nn
import torch


class MultiHeadCrossEntropyLoss(nn.Module):
    def __init__(self, num_head:int, weight_per_loss:Optional[list]=None, **kwargs):
        super(MultiHeadCrossEntropyLoss, self).__init__()
        self.num_head = num_head
        self.loss_fn = [nn.CrossEntropyLoss(**kwargs)] * num_head

        if weight_per_loss is None:
            self.weight_per_loss = [1/num_head]*num_head
        else:
            if len(weight_per_loss) == num_head:
                self.weight_per_loss = weight_per_loss
            else:
                raise ValueError('The length of the weight per loss should be same as num_head')
    

    def forward(self, predict:List[torch.Tensor], target:List[torch.Tensor]) -> torch.Tensor:
        result = 0
        for i in range(self.num_head):
            loss = self.loss_fn[i](predict[i], target[i])
            result += loss*self.weight_per_loss[i]
        
        return result

if __name__ == '__main__':
    loss = MultiHeadCrossEntropyLoss(3)
    a = [torch.zeros(1,20, dtype=torch.float32), torch.zeros(1,20, dtype=torch.float32), torch.zeros(1,10, dtype=torch.float32)]
    b = [torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0])]

    losses = loss(a,b)

    print(losses.item())

    a = [torch.zeros(1,5, dtype=torch.float32), torch.zeros(1,5, dtype=torch.float32), torch.zeros(1,5, dtype=torch.float32)]
    b = [torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0])]

    losses = loss(a,b)

    print(losses.item())