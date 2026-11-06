
import torch
from loss_function import prediction_loss, fast_penalty, slow_penalty

def test_prediction_loss():
    prediction = torch.tensor([[0.8, 0.2]])
    target = torch.tensor([[1.0, 0.0]])

    loss = prediction_loss(prediction, target)

    assert torch.isfinite(loss)
    print("Prediction loss:", loss.item())

def test_fast_penalty():
    states = torch.randn(2, 5, 8)
    penalty = fast_penalty(states)

    assert torch.isfinite(penalty)
    assert penalty.item() >= 0

def test_slow_penalty():
    states = torch.randn(2, 5, 8)
    penalty = slow_penalty(states)

    assert torch.isfinite(penalty)
    assert penalty.item() >= 0

if __name__ == "__main__":
    test_prediction_loss()
    test_fast_penalty()
    test_slow_penalty()
    print("All tests passed!")

#Prediction loss: 0.2231435477733612
#All tests passed!