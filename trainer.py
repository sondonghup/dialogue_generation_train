import wandb
import time
import os
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DGTrainer:
    def __init__(self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        num_epochs,
        device,
        tokenizer,
        gradient_clip_val: float,
        accumulate_grad_batches: int,
        log_every: int = 20,
        save_every: int = 10_000,
        save_dir: str = 'ckpt'
        ):
        super(DGTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.tokenizer = tokenizer
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_every = log_every
        self.save_every = save_every
        self.save_dir = save_dir

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def _train_epoch(self):
        total_loss = 0.0

        self.model.train()

        # print(f'model_config : {self.model}')

        for step, (input_ids, attention_masks, targets) in enumerate(tqdm(self.train_loader)):
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            targets = targets.to(self.device)

            # print(f'\n\ninput_ids : {input_ids.size()}')
            # print(f'attention_masks : {attention_masks.size()}')
            # print(f'targets : {targets.size()}')

            # print(f'input_ids : {input_ids[0]}')
            # print(f'attention_masks : {attention_masks[0]}')
            # print(f'targets : {targets[0]}')

            torch.autograd.set_detect_anomaly(True)

            outputs = self.model(input_ids = input_ids, attention_mask = attention_masks, labels = targets)
            loss = outputs.loss / self.accumulate_grad_batches
            # print(f'\noutputs.loss : {outputs.loss}')
            # print(f'self.accumulate_grad_batches : {self.accumulate_grad_batches}')
            # print(f'loss : {loss}')
            self.optimizer.zero_grad()
            loss.backward()
            
            # print(f'loss.item : {loss.item}')
            total_loss += loss.item()
            
            # print(f'total_loss : {total_loss}')

            if step % self.accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
                self.scheduler.step()

            if step % self.log_every == 0:
                train_mean_loss = total_loss / (step + 1)
                
                # print(f'train_mean_loss : {train_mean_loss}')
                wandb.log({
                    "lr" : self._get_lr(),
                    "train_loss" : train_mean_loss,
                    "batch_size" : input_ids.size(0)
                })
            
            if step % self.save_every == 0:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                date_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

                os.mkdir(os.path.join(self.save_dir, date_time))
                self.model.save_pretrained(os.path.join(self.save_dir, date_time, 'model.pt'))
                # torch.save(self.model, os.path.join(self.save_dir, date_time, 'model.pt'))
                # self.model.to_json_file(os.path.join(self.save_dir, date_time, 'config.json'))

    def _validate(self, epoch):
        total_loss = 0.0

        self.model.eval()

        with torch.no_grad():
            for step, (input_ids, attention_masks, targets) in enumerate(tqdm(self.valid_loader)):
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(input_ids = input_ids, attention_mask = attention_masks, labels = targets)
                loss = outputs.loss

                total_loss += loss.item()

        val_mean_loss = total_loss / len(self.valid_loader)

        wandb.log({
            "valid_loss" : val_mean_loss
        })

    def fit(self):
        print(f'num_epochs : {self.num_epochs}')
        for epoch in range(self.num_epochs):
            logger.info(f"epoch {epoch} start")
            self._train_epoch()

            logger.info("validate")
            self._validate(epoch)