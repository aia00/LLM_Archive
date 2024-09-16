from collections import OrderedDict
import re
import os
import sys

sys.path.insert(0, '../')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run, get_parrallel_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names


run_dir = "/home/ykwang/projects/LLM_Archive/GPT2/experiments/ICL_24_2/mycode/categorical_loss/models"


model_name_without_label = "multiple_task_without_label"
# model_name_without_label = "multiple_task_without_label"
task_without_label = "multiple_task_without_label"

run_id_euqal_weight = "equal_noisy"  # if you train more models, replace with the run_id from the table above
run_id_1over6_weight = "1over24_noisy"
run_id_only_linear = "only_linear"

run_path_without_label_equal_weight = os.path.join(run_dir, model_name_without_label, run_id_euqal_weight)
run_path_without_label_1over6_weight = os.path.join(run_dir, model_name_without_label, run_id_1over6_weight)
run_path_without_label_only_linear = os.path.join(run_dir, model_name_without_label, run_id_only_linear)



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from samplers import get_data_sampler
from tasks import get_task_sampler

model_without_label_euqal_weight, conf = get_parrallel_model_from_run(run_path_without_label_equal_weight)
n_dims = conf.model.n_dims
batch_size = conf.training.batch_size  * 40

data_sampler = get_data_sampler(conf.training.data, n_dims)
task_sampler = get_task_sampler(
    "noisy_linear_regression",
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)

task = task_sampler()
xs= data_sampler.sample_xs(b_size=batch_size, n_points=40).to(device)
ys = task.evaluate(xs).to(device)




model_without_label_1over6_weight, conf = get_parrallel_model_from_run(run_path_without_label_1over6_weight)

model_without_label_1over6_weight = model_without_label_1over6_weight.to(device)

with torch.no_grad():
    pred_without_label = model_without_label_1over6_weight(xs, ys)

metric_without_label = task.get_metric()
loss_without_label_1over6_weight = metric_without_label(pred_without_label, ys).cpu().numpy()




model_without_label_equal_weight, conf = get_parrallel_model_from_run(run_path_without_label_equal_weight)
model_without_label_equal_weight = model_without_label_equal_weight.to(device)

with torch.no_grad():
    pred_without_label = model_without_label_equal_weight(xs, ys)

metric_without_label = task.get_metric()
loss_without_label_equal_weight = metric_without_label(pred_without_label, ys).cpu().numpy()


model_without_label_only_linear, conf = get_parrallel_model_from_run(run_path_without_label_only_linear)
model_without_label_only_linear = model_without_label_only_linear.to(device)

with torch.no_grad():
    pred_without_label = model_without_label_only_linear(xs, ys)

metric_without_label = task.get_metric()
loss_without_label_only_linear = metric_without_label(pred_without_label, ys).cpu().numpy()



plt.plot(loss_without_label_1over6_weight.mean(axis=0), lw=2, label="multi task unbalanced")
plt.plot(loss_without_label_equal_weight.mean(axis=0), lw=2, label="multi task")
plt.plot(loss_without_label_only_linear.mean(axis=0), lw=2, label="single task")
# plt.plot(loss_without_label_cat_loss.mean(axis=0), lw=2, label="without_label_cat_loss")

plt.xlabel("# Linear_Regression in-context examples")
plt.ylabel("squared error")
plt.legend()
plt.savefig('plot.png')

