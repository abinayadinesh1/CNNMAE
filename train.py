from torch import nn
from torch import optim

from torchvision import transforms

from model import FCMAE
from config import Config
from custom_dataset import MaskedDataset


# dataset/dataloader object declaration
data = MaskedDataset("data", (256, 256), 16, 1)

# model object declaration
config = Config([3, 16, 32, 64])
model = FCMAE(config)

# optimizer, loss object declaration
mse_loss = nn.MSELoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr = 0.001) # pass in weights so adam knows what to change

# training loop
EPOCH = 51
for epoch in range(EPOCH):
    print("EPOCH: " + str(epoch))
    for sample in data:
        unmasked_image = sample["unmasked"]
        masked_image = sample["masked"]

        masked_regions = 1-(masked_image==unmasked_image).long()

        output = model(masked_image)
        actual_comparison_output = output*masked_regions
        actual_comparison_target = output*unmasked_image

        optimizer.zero_grad()
        loss = mse_loss(actual_comparison_output, actual_comparison_target)
        loss.backward()
        optimizer.step()

        print("Loss: " + str(loss))

        if epoch % 10 == 0:
            to_PIL = transforms.ToPILImage()
            saved = to_PIL(output[0])
            saved.save("output/" + str(epoch) + ".jpg")
    