import torch
from advanced_model import AdvancedRoadModel
from loss import DiceBCELoss

def test_architecture():
    print("Testing Model Architecture...")
    model = AdvancedRoadModel(num_classes=1)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (1, 1, 256, 256), f"Expected output (1, 1, 256, 256), got {y.shape}"
    print("✅ Model Forward Pass Successful")

def test_loss():
    print("\nTesting Loss Function...")
    criterion = DiceBCELoss()
    pred = torch.randn(1, 1, 256, 256, requires_grad=True)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    
    loss = criterion(pred, target)
    print(f"Loss value: {loss.item()}")
    
    loss.backward()
    print("✅ Loss Backward Pass Successful")

if __name__ == "__main__":
    try:
        test_architecture()
        test_loss()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
