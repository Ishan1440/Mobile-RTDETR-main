'''
by lyuwenyu
'''
import time
import json
import datetime

import torch
from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
# import torch_tensorrt
import gc
from fvcore.nn import flop_count_table, FlopCountAnalysis

class DetSolver(BaseSolver):

    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(
            p.numel()
            for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        '''Counts how many parameters (trainable numbers) are in the model. Useful for reporting model size.'''

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        '''Wraps the validation dataset into COCO API format (standard in object detection).'''
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }
        '''Initializes best_stat dict to track best evaluation results.'''
        # for g in self.optimizer.param_groups:
        #     g['lr'] = 1e-5

        # self.model = torch.compile(self.model,backend="torch_tensorrt",dynamic=False)
        start_time = time.time()

        for epoch in range(self.last_epoch + 1, args.epoches):
            torch.cuda.empty_cache() #clears GPU cache
            gc.collect() #Python's garbage collector to free memory

            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            '''Ensures each GPU gets a different data shard each epoch (for distributed training).'''

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)
            '''Train for one epoch, runs forward + backward passes, optimizer step, EMA updates, gradient scaling. Returns training stats.'''

            self.lr_scheduler.step()
            '''Updates the learning rate according to the scheduler policy'''

            if self.output_dir:
                checkpoints_dir = self.output_dir / 'checkpoints'
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_paths = [checkpoints_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                '''didn't understand the above comment!'''
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(
                        checkpoints_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(
                        self.state_dict(epoch), checkpoint_path)
            '''Saves the training state (checkpoint) for every epoch'''

            module = self.ema.module if self.ema else self.model
            # record previous best bbox AP to detect improvement this epoch
            prev_best_bbox_ap = best_stat.get('coco_eval_bbox', float('-inf'))
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )
            '''Validation'''

            # TODO
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            '''Updates best_stat with the best epoch'''
            print('best_stat: ', best_stat)

            # Save best checkpoint when bbox AP improves
            if self.output_dir and 'coco_eval_bbox' in test_stats and best_stat.get('coco_eval_bbox', float('-inf')) > prev_best_bbox_ap:
                checkpoints_dir = self.output_dir / 'checkpoints'
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                best_ckpt_path = checkpoints_dir / 'best.pth'
                dist.save_on_master(self.state_dict(epoch), best_ckpt_path)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                '''Only the main process to write to log.txt'''

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name)
                '''Saves COCO evaluation metrics (like bounding box results), every epoch (latest.pth) and also every 50th epoch.'''

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def show_flops(self, input=None):
        self.setup()
        self.eval()
        self.model.eval()
        if input is None:
            input = torch.randn(1, 3, 640, 640).to(self.cfg.device)
        flops = FlopCountAnalysis(self.model, input)
        print(flop_count_table(flops,max_depth=2))
       
    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor,
            self.val_dataloader, base_ds, self.device, self.output_dir)

        if self.output_dir:
            dist.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
