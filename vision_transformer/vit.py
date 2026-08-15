import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.utils.data as dataloader
import torch.nn as nn

#VIT BASED ON MNIST 10 classes

DATASET_PATH = '../../dataset/mnist'

#transformation of PIL into tensor format
transformation_operation = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=True, download=True, transform=transformation_operation)
val_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=False, download=True, transform=transformation_operation)



#define variables
batch_size = 64
num_classes = 10
num_channels = 1 #number of channels in the image
img_size = 28 #size of the image
patch_size = 7 #size of the patch
patch_num = (img_size//patch_size)**2
attention_heads = 4 #number of attention heads
embed_dim = 64 #embedding dimension for the patch embeddings
transformer_blocks = 4 #number of transformer blocks
mlp_nodes = 128 #number of nodes in the MLP
learning_rate = 3e-4
epochs = 5


train_loader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader  = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# class for patch embedding - Part 1 of the ViT
class PatchEmbedding(nn.Module):
  def __init__(self):
    super().__init__()
    self.patch_embed = nn.Conv2d(num_channels, embed_dim, kernel_size = patch_size, stride = patch_size)

  def forward(self, x):
    x = self.patch_embed(x)
    x = x.flatten(2)
    x = x.transpose(1,2)
    return x


# class for the transformer blocks - Part 2 of the ViT
# class for transformer encoder - Part 2 of the ViT
class TransformerEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(embed_dim)
    self.multi_head_attention = nn.MultiheadAttention(embed_dim, attention_heads, batch_first=True)
    self.layer_norm2 = nn.LayerNorm(embed_dim)
    self.mlp = nn.Sequential(
        nn.Linear(embed_dim, mlp_nodes),
        nn.GELU(),
        nn.Linear(mlp_nodes, embed_dim)
    )

  def forward(self, x):
    residual1 = x
    x = self.layer_norm1(x)
    x = self.multi_head_attention(x, x, x)[0]
    x = x + residual1

    residual2 = x
    x = self.layer_norm2(x)
    x = self.mlp(x)
    x = x + residual2

    return x

# MLP head Part 3 of the ViT
class MLP_Head(nn.Module):
  def __init__(self):
    super().__init__()
    self.layernorm1 = nn.LayerNorm(embed_dim)
    self.mlp_head = nn.Linear(embed_dim, num_classes)

  def forward(self, x):
    # x = x[:,0]
    x = self.layernorm1(x)
    x = self.mlp_head(x)
    return x

class VisionTransformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.patch_embedding = PatchEmbedding()
    self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
    self.position_embedding = nn.Parameter(torch.randn(1,patch_num+1, embed_dim))
    self.transformer_blocks = nn.Sequential(*[TransformerEncoder() for _ in range(transformer_blocks)])
    self.mlp_head = MLP_Head()

  def forward(self, x):
    x = self.patch_embedding(x)
    B = x.size(0)
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim = 1)
    x = x + self.position_embedding
    x = self.transformer_blocks(x)
    x = x[:,0]
    x = self.mlp_head(x)
    return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
print(f"Model on {device} device")

for epoch in range(5):
    model.train()
    total_loss = 0
    correct_epoch = 0
    total_epoch = 0
    print(f"\nEpoch {epoch+1}")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)

        correct_epoch += correct
        total_epoch += labels.size(0)

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx+1:3d}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%")

    epoch_acc = 100.0 * correct_epoch / total_epoch
    print(f"==> Epoch {epoch+1} Summary: Total Loss = {total_loss:.4f}, Accuracy = {epoch_acc:.2f}%")


    # Switch to evaluation mode
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = 100.0 * correct / total
print(f"\n==> Val Accuracy: {test_acc:.2f}%")


import matplotlib.pyplot as plt

# Show 10 predictions from the first test batch
model.eval()
images, labels = next(iter(val_loader))
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(dim=1)

# Move to CPU for plotting
images = images.cpu()
preds = preds.cpu()
labels = labels.cpu()

# Plot first 10 images
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}")
    plt.axis('off')
plt.tight_layout()
output_path = "predictions.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved predictions plot to {output_path}")