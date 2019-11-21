import torch
import ignite
import torch_helpers


def get_trainer(model, criterion, optimizer, config):
    
    def train_step(engine, batch):

        model.train()

        if engine.state.iteration % config['accumulation_steps'] == 0:
            optimizer.zero_grad()

        batch = torch_helpers.batch_to_model_device(batch, model)
        if 'mixup_alpha' in config and config['mixup_alpha'] > 0:
            batch = torch_helpers.mixup_batch(batch, alpha=config['mixup_alpha'])
        logits = model(batch['features'])
        loss = criterion(logits, batch['targets']) / config['accumulation_steps']
        loss.backward()

        if engine.state.iteration % config['accumulation_steps'] == 0:
            if config['clip_norm']:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['clip_norm']
                )
            optimizer.step()

        return loss.item() * config['accumulation_steps']
       
    ignite.engine.Engine(train_step)
    ignite.metrics.RunningAverage(
        output_transform=lambda x: x, alpha=0.98
    ).attach(trainer, 'running avg loss')
    
    return trainer
