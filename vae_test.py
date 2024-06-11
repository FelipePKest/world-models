import torch
from models import VAE
from torchvision import transforms
from utils.misc import RolloutGenerator

# Hardcoded for now
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

rollout = RolloutGenerator("exp_dir")
obs = rollout.env.reset()
rollout.env.render()
hidden = [
    torch.zeros(1, RSIZE).to(rollout.device)
    for _ in range(2)]

cumulative = 0
i = 0

obs = transform(obs[0]).unsqueeze(0).to(rollout.device)
_, latent_mu, _ = rollout.vae(obs)
action = rollout.controller(latent_mu, hidden[0])
_, _, _, _, _, next_hidden = rollout.mdrnn(action, latent_mu, hidden)
print(latent_mu)
# action.squeeze().cpu().detach().numpy()
act, hid = action.squeeze().cpu().detach().numpy(), next_hidden
# obs, reward, done, a, b = rollout.env.step(action)
# cumulative += reward
# print(-cumulative)
i += 1


print(act)