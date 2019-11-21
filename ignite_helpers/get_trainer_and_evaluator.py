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
    
    return train_step

def get_evaluator(model):
    def evaluate_step(engine, batch):

        model.train()
        batch = torch_helpers.batch_to_model_device(batch, model)
        with torch.no_grad():
            output = model(batch['features'])

        return dict(
            output=output,
            targets=batch['targets'],
        )
    
    return evaluate_step


def get_trainer_and_evaluator(model, criterion, optimizer, config):
    trainer = ignite.engine.Engine(get_trainer(model, criterion, optimizer, config))
    evaluator = ignite.engine.Engine(get_evaluator(model))

    ignite.metrics.RunningAverage(
        output_transform=lambda x: x, alpha=0.98
    ).attach(trainer, 'running avg loss')

    return trainer, evaluator
