from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

def epoch_ssl(encoder, projector, opt_encoder, opt_projector, dataloader, device):
    mse_loss = torch.nn.MSELoss()
    encoder.train()
    projector.train()
    for batch1, batch2 in tqdm(dataloader):
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        opt_encoder.zero_grad()
        opt_projector.zero_grad()

        representation1 = encoder(batch1).squeeze()
        representation2 = encoder(batch2).squeeze()

        projection1 = projector(representation1)
        projection2 = projector(representation2)

        # invariance loss
        inv_loss = mse_loss(projection1, projection2)

        # variance loss
        std_z_a = torch.sqrt(projection1.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(projection2.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # covariance loss
        N, D = projection1.shape
        z_a = projection1 - projection1.mean(dim=0)
        z_b = projection2 - projection2.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)
        cov_loss = (cov_z_a**2).sum() / D + (cov_z_b**2).sum() / D
        diag_loss = torch.diagonal(cov_z_a**2).sum()/D - torch.diagonal(cov_z_b**2).sum()/D
        cov_loss -= diag_loss
        # loss

        gamma, mu, nu = 15.0, 25.0, 1.0
        loss = gamma * inv_loss + mu * std_loss + nu * cov_loss

        loss.backward()
        opt_encoder.step()
        opt_projector.step()

def epoch_supervised(encoder, head, opt_head, dataloader, device):
    head.train()
    encoder.eval()
    for batch, target in tqdm(dataloader):
        batch = batch.to(device)
        target = target.to(device)

        opt_head.zero_grad()

        with torch.no_grad():
            representation = encoder(batch).squeeze()
        x = head(representation)
        classif = F.log_softmax(x, dim=1)

        loss = F.nll_loss(classif, target)
        loss.backward()
        opt_head.step()

def test_supervised(encoder, head, dataloader, device):
    encoder.eval()
    head.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            representation = encoder(data).squeeze()
            x = head(representation)
            output = F.log_softmax(x, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))
