import os
import pdb
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import torch.nn as nn
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import numpy as np

import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, cat):
    optimizer.zero_grad()
    # pdb.set_trace()
    if cat[0] is not None:
        if cat[1] == False:
            output = model(xs, ys, cat[0])
            loss = loss_func(output, ys)
        else:
            output = model(xs, ys, cat[0], cat[1])
            loss = loss_func(output, ys, cat[0])
    else:
        output = model(xs, ys)
        loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    if cat[0] is not None and cat[1]==True:
        return loss.detach().item(), output[0].detach()
    else:
        return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    # task_sampler = get_task_sampler(
    #     args.training.task,
    #     n_dims,
    #     bsize,
    #     num_tasks=args.training.num_tasks,
    #     **args.training.task_kwargs,
    # )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    # task_choices = {0:'noisy_linear_regression', 1:"quadratic_regression", 2:"cube_regression",
                    # 3:"relu_2nn_regression", 4: "decision_tree"} 
    # task_choices = {0:'linear_regression', 1:"quadratic_regression", 2:"cube_regression",
    #                 3:"relu_2nn_regression", 4: "decision_tree"} 
    # task_choices = {3:"relu_2nn_regression",} 
    # task_choices = {2:"cube_regression",} 
    # task_choices = {0:"quadratic_regression",1:"decision_tree"} 
    # task_choices = {0:"quadratic_regression"}
    task_choices = {0:"quadratic_regression",1:"linear_classification"}
    # task_choices = {0:'linear_classification', 1:"quadratic_regression", 2:"cube_regression",
    #                 3:"relu_2nn_regression", 4: "decision_tree"} 
    task_choices_keys = list(task_choices.keys())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        if args.training.task in ["multiple_task_with_label", "multiple_task_with_label_cat_loss","multiple_task_without_label_cat_loss"]:
            task_key = np.random.choice(task_choices_keys)
            task_type = task_choices[task_key]
            cat_num = len(task_choices_keys)

        elif args.training.task == "multiple_task_without_label":
            task_key = np.random.choice(task_choices_keys)
            task_type = task_choices[task_key]
            task_key = None
            cat_num = len(task_choices_keys)

        else:
            task_type = args.training.task
            task_key = None
            cat_num = None


        task_sampler = get_task_sampler(
                task_type,
                n_dims,
                bsize,
                num_tasks=args.training.num_tasks,
                **args.training.task_kwargs, )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)


        
        if args.training.task in ["multiple_task_with_label_cat_loss", "multiple_task_without_label_cat_loss"]:
            loss_func = task.get_training_metric_with_cat_loss()
            # loss_func = task.get_training_metric()
            cat_loss_bool = True
        else:
            loss_func = task.get_training_metric()
            cat_loss_bool = False
        
        xs = xs.to(device)
        ys = ys.to(device)
       
        
        loss, output = train_step(model, xs, ys, optimizer, loss_func, [task_key,cat_loss_bool])

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    # model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "gpt2_labeled", "gpt2_labeled_cat"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
