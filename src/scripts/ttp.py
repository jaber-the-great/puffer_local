#!/usr/bin/env python3

'''
This script is designed to either train a new model or retrain existing models (via continual learning), 
using time-series data collected from streaming sessions, and optimizes the model to predict transmission 
times based on various network conditions.
'''

import sys
import json
import argparse
import yaml
import torch
from os import path
from datetime import datetime, timedelta
import numpy as np
from multiprocessing import Process
import gc
import pandas as pd 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from helpers import (
    connect_to_influxdb, connect_to_postgres,
    make_sure_path_exists, retrieve_expt_config, create_time_clause,
    get_expt_id, get_user)


VIDEO_DURATION = 180180
PKT_BYTES = 1500
MILLION = 1000000

# training related
BATCH_SIZE = 32
NUM_EPOCHS = 500
CHECKPOINT = 100

CL_MAX_DATA_SIZE = 1000000  # 1 million rows of data
CL_DISCOUNT = 0.9  # sampling weight discount
CL_MAX_DAYS = 14  # sample from last 14 days

TUNING = False
DEVICE = torch.device('cpu')

# cache of Postgres data: experiment 'id' -> json 'data' of the experiment
expt_id_cache = {}


class Model:
    PAST_CHUNKS = 8
    FUTURE_CHUNKS = 5
    DIM_IN = 62
    BIN_SIZE = 0.5  # seconds
    BIN_MAX = 20
    DIM_OUT = BIN_MAX + 1
    DIM_H1 = 64
    DIM_H2 = 64
    WEIGHT_DECAY = 1e-4
    LEARNING_RATE = 1e-4

    def __init__(self, model_path=None):
        # define model, loss function, and optimizer
        self.model = torch.nn.Sequential(
            torch.nn.Linear(Model.DIM_IN, Model.DIM_H1),
            torch.nn.ReLU(),
            torch.nn.Linear(Model.DIM_H1, Model.DIM_H2),
            torch.nn.ReLU(),
            torch.nn.Linear(Model.DIM_H2, Model.DIM_OUT),
        ).double().to(device=DEVICE)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device=DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=Model.LEARNING_RATE,
                                          weight_decay=Model.WEIGHT_DECAY)

        self.obs_size = None
        self.obs_mean = None
        self.obs_std = None

    def set_model_train(self):
        self.model.train()

    def set_model_eval(self):
        self.model.eval()

    def update_obs_stats(self, raw_in):
        if self.obs_size is None:
            self.obs_size = len(raw_in)
            self.obs_mean = np.mean(raw_in, axis=0)
            self.obs_std = np.std(raw_in, axis=0)
            return

        # update population size
        old_size = self.obs_size
        new_size = len(raw_in)
        self.obs_size = old_size + new_size

        # update popultation mean
        old_mean = self.obs_mean
        new_mean = np.mean(raw_in, axis=0)
        self.obs_mean = (old_mean * old_size + new_mean * new_size) / self.obs_size

        # update popultation std
        old_std = self.obs_std
        old_sum_square = old_size * (np.square(old_std) + np.square(old_mean))
        new_sum_square = np.sum(np.square(raw_in), axis=0)
        mean_square = (old_sum_square + new_sum_square) / self.obs_size
        self.obs_std = np.sqrt(mean_square - np.square(self.obs_mean))

    def normalize_input(self, raw_in, update_obs=False):
        z = np.array(raw_in)

        # update mean and std of the data seen so far
        if update_obs:
            self.update_obs_stats(z)

        assert(self.obs_size is not None)

        for col in range(len(self.obs_mean)):
            z[:, col] -= self.obs_mean[col]
            if self.obs_std[col] != 0:
                z[:, col] /= self.obs_std[col]

        return z

    # special discretization: [0, 0.5 * BIN_SIZE)
    # [0.5 * BIN_SIZE, 1.5 * BIN_SIZE), [1.5 * BIN_SIZE, 2.5 * BIN_SIZE), ...
    def discretize_output(self, raw_out):
        z = np.array(raw_out)

        z = np.floor((z + 0.5 * Model.BIN_SIZE) / Model.BIN_SIZE).astype(int)
        return np.clip(z, 0, Model.BIN_MAX)

    # perform one step of training (forward + backward + optimize)
    def train_step(self, input_data, output_data):
        x = torch.from_numpy(input_data).to(device=DEVICE)
        y = torch.from_numpy(output_data).to(device=DEVICE)

        # forward pass
        y_scores = self.model(x)
        loss = self.loss_fn(y_scores, y)

        # backpropagation and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # compute loss
    def compute_loss(self, input_data, output_data):
        with torch.no_grad():
            x = torch.from_numpy(input_data).to(device=DEVICE)
            y = torch.from_numpy(output_data).to(device=DEVICE)

            y_scores = self.model(x)
            loss = self.loss_fn(y_scores, y)

            return loss.item()

    # compute accuracy of the classifier
    def compute_accuracy(self, input_data, output_data):
        correct = 0
        total = 0

        with torch.no_grad():
            x = torch.from_numpy(input_data).to(device=DEVICE)
            y = torch.from_numpy(output_data).to(device=DEVICE)

            y_scores = self.model(x)
            y_predicted = torch.max(y_scores, 1)[1].to(device=DEVICE)

            total += y.size(0)
            correct += (y_predicted == y).sum().item()

        return correct / total

    # makes predictions on input data and maps the output to the discretized bin values
    def predict(self, input_data):
        with torch.no_grad():
            x = torch.from_numpy(input_data).to(device=DEVICE)

            y_scores = self.model(x)
            y_predicted = torch.max(y_scores, 1)[1].to(device=DEVICE)

            ret = y_predicted.double().numpy()
            for i in range(len(ret)):
                bin_id = ret[i]
                if bin_id == 0:  # the first bin is defined differently
                    ret[i] = 0.25 * Model.BIN_SIZE
                else:
                    ret[i] = bin_id * Model.BIN_SIZE

            return ret

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.obs_size = checkpoint['obs_size']
        self.obs_mean = checkpoint['obs_mean']
        self.obs_std = checkpoint['obs_std']

    def save(self, model_path):
        assert(self.obs_size is not None)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'obs_size': self.obs_size,
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
        }, model_path)

    def save_cpp_model(self, model_path, meta_path):
        # save model to model_path
        example = torch.rand(1, Model.DIM_IN).double()
        traced_script_module = torch.jit.trace(self.model, example)
        traced_script_module.save(model_path)

        # save obs_size, obs_mean, obs_std to meta_path
        meta = {'obs_size': self.obs_size,
                'obs_mean': self.obs_mean.tolist(),
                'obs_std': self.obs_std.tolist()}
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh)


# Validates command-line arguments to ensure correct inputs and parameters are provided. It checks if model directories exist,
# if inference/tuning options are used correctly, and if GPU is available when required.
def check_args(args):
    if args.load_model:
        if not path.isdir(args.load_model):
            sys.exit('Error: directory {} does not exist'
                     .format(args.load_model))

        for i in range(Model.FUTURE_CHUNKS):
            model_path = path.join(args.load_model, 'py-{}.pt'.format(i))
            if not path.isfile(model_path):
                sys.exit('Error: Python model {} does not exist'
                         .format(model_path))

    if args.save_model:
        make_sure_path_exists(args.save_model)

        for i in range(Model.FUTURE_CHUNKS):
            model_path = path.join(args.save_model, 'py-{}.pt'.format(i))
            if path.isfile(model_path):
                sys.exit('Error: Python model {} already exists'
                         .format(model_path))

            model_path = path.join(args.save_model, 'cpp-{}.pt'.format(i))
            if path.isfile(model_path):
                sys.exit('Error: C++ model {} already exists'
                         .format(model_path))

            meta_path = path.join(args.save_model, 'cpp-meta-{}.pt'.format(i))
            if path.isfile(meta_path):
                sys.exit('Error: meta {} already exists'.format(meta_path))

    if args.inference:
        if not args.load_model:
            sys.exit('Error: need to load model before inference')

        if args.tune or args.save_model:
            sys.exit('Error: cannot tune or save model during inference')
    else:
        if not args.save_model:
            sys.exit('Error: specify a folder to save models\n')

    # want to tune hyperparameters
    if args.tune:
        if args.save_model:
            sys.stderr.write('Warning: model would better be trained with '
                             'validation dataset\n')

        global TUNING
        TUNING = True

    # set device to CPU or GPU
    if args.enable_gpu:
        if not torch.cuda.is_available():
            sys.exit('Error: --enable-gpu is set but no CUDA is available')

        global DEVICE
        DEVICE = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

    # continual learning
    if args.cl:
        if not args.load_model or not args.save_model:
            sys.exit('Error: pass --load-model and --save-model to perform '
                     'continual learning')

        if args.time_start or args.time_end:
            sys.exit('Error: --cl conflicts with --from and --to; it has its '
                     'own strategy to sample data from specific durations')

        if args.inference:
            sys.exit('Error: cannot perform inference with --cl turned on')

        # reduce number of epochs if training on a previous model
        global NUM_EPOCHS
        NUM_EPOCHS = 300


# This function computes the transmission time (trans_time) for video chunks by comparing the time they were sent (sent_ts)
# with the time they were acknowledged (acked_ts). It also extracts relevant TCP metrics like delivery_rate, cwnd, rtt, etc.
def calculate_trans_times(video_sent_results, video_acked_results,
                          cc, postgres_cursor):
    d = {}
    last_video_ts = {}

    for pt in video_sent_results['video_sent']:
        expt_id = get_expt_id(pt)
        session = (get_user(pt), int(pt['init_id']),
                   pt['channel'], expt_id)

        # filter data points by congestion control
        expt_config = retrieve_expt_config(expt_id, expt_id_cache,
                                           postgres_cursor)
        if cc is not None and expt_config['cc'] != cc:
            continue

        if session not in d:
            d[session] = {}
            last_video_ts[session] = None

        video_ts = int(pt['video_ts'])

        if last_video_ts[session] is not None:
            if video_ts != last_video_ts[session] + VIDEO_DURATION:
                continue

        last_video_ts[session] = video_ts

        d[session][video_ts] = {}
        dsv = d[session][video_ts]  # short name

        dsv['sent_ts'] = np.datetime64(pt['time'])
        dsv['size'] = float(pt['size']) / PKT_BYTES  # bytes -> packets
        # byte/second -> packet/second
        dsv['delivery_rate'] = float(pt['delivery_rate']) / PKT_BYTES
        dsv['cwnd'] = float(pt['cwnd'])
        dsv['in_flight'] = float(pt['in_flight'])
        dsv['min_rtt'] = float(pt['min_rtt']) / MILLION  # us -> s
        dsv['rtt'] = float(pt['rtt']) / MILLION  # us -> s

    for pt in video_acked_results['video_acked']:
        expt_id = get_expt_id(pt)
        session = (get_user(pt), int(pt['init_id']),
                   pt['channel'], expt_id)

        # filter data points by congestion control
        expt_config = retrieve_expt_config(expt_id, expt_id_cache,
                                           postgres_cursor)
        if cc is not None and expt_config['cc'] != cc:
            continue

        if session not in d:
            continue

        video_ts = int(pt['video_ts'])
        if video_ts not in d[session]:
            continue

        dsv = d[session][video_ts]  # short name

        # calculate transmission time
        sent_ts = dsv['sent_ts']
        acked_ts = np.datetime64(pt['time'])
        dsv['acked_ts'] = acked_ts
        dsv['trans_time'] = (acked_ts - sent_ts) / np.timedelta64(1, 's')
    
    # print the first 1 records of the dictionary
    # for i, (k, v) in enumerate(d.items()):
    #     if i == 1:
    #         break
    #     print(k, v)

    return d


# This function retrieves data from InfluxDB and PostgreSQL, using time-based filters if provided, 
# and then processes it into structured form with the help of calculate_trans_times().
def prepare_raw_data(yaml_settings_path, time_start, time_end, cc):
    with open(yaml_settings_path, 'r') as fh:
        yaml_settings = yaml.safe_load(fh)

    # construct time clause after 'WHERE'
    time_clause = create_time_clause(time_start, time_end)

    # create a client connected to InfluxDB
    influx_client = connect_to_influxdb(yaml_settings)

    # perform queries in InfluxDB
    video_sent_query = 'SELECT * FROM video_sent'
    if time_clause is not None:
        video_sent_query += ' WHERE ' + time_clause
    video_sent_results = influx_client.query(video_sent_query)
    if not video_sent_results:
        sys.stderr.write('Warning: no results returned from query: '
                         + video_sent_query)
        return None

    video_acked_query = 'SELECT * FROM video_acked'
    if time_clause is not None:
        video_acked_query += ' WHERE ' + time_clause
    video_acked_results = influx_client.query(video_acked_query)
    if not video_acked_results:
        sys.stderr.write('Warning: no results returned from query: '
                         + video_acked_query)
        return None

    # create a client connected to Postgres
    postgres_client = connect_to_postgres(yaml_settings)
    postgres_cursor = postgres_client.cursor()

    # calculate chunk transmission times
    ret = calculate_trans_times(video_sent_results, video_acked_results,
                                cc, postgres_cursor)

    postgres_cursor.close()
    return ret


# Appends historical chunk data, including features like delivery_rate, cwnd, and trans_time, 
# to the input vector. If past data is missing, it pads the input vector with the nearest available chunk data.
def append_past_chunks(ds, next_ts, row):
    i = 1
    past_chunks = []

    while i <= Model.PAST_CHUNKS:
        ts = next_ts - i * VIDEO_DURATION
        if ts in ds and 'trans_time' in ds[ts]:
            past_chunks = [ds[ts]['delivery_rate'],
                           ds[ts]['cwnd'], ds[ts]['in_flight'],
                           ds[ts]['min_rtt'], ds[ts]['rtt'],
                           ds[ts]['size'], ds[ts]['trans_time']] + past_chunks
        else:
            nts = ts + VIDEO_DURATION  # padding with the nearest ts
            padding = [ds[nts]['delivery_rate'],
                       ds[nts]['cwnd'], ds[nts]['in_flight'],
                       ds[nts]['min_rtt'], ds[nts]['rtt']]

            if nts == next_ts:
                padding += [0, 0]  # next_ts is the first chunk to send
            else:
                padding += [ds[nts]['size'], ds[nts]['trans_time']]

            break

        i += 1

    if i != Model.PAST_CHUNKS + 1:  # break in the middle; padding must exist
        while i <= Model.PAST_CHUNKS:
            past_chunks = padding + past_chunks
            i += 1

    row += past_chunks


# Converts the processed raw data into input-output pairs for the model. It uses append_past_chunks() to construct the input vectors
# and appends TCP information like delivery_rate, rtt, and trans_time.
# return FUTURE_CHUNKS pairs of (raw_in, raw_out)
def prepare_input_output(d):
    ret = [{'in':[], 'out':[]} for _ in range(Model.FUTURE_CHUNKS)]

    for session in d:
        ds = d[session]

        for next_ts in ds:
            if 'trans_time' not in ds[next_ts]:
                continue

            # construct a single row of input data
            row = []

            # append past chunks with padding
            append_past_chunks(ds, next_ts, row)

            # append the TCP info of the next chunk
            row += [ds[next_ts]['delivery_rate'],
                    ds[next_ts]['cwnd'], ds[next_ts]['in_flight'],
                    ds[next_ts]['min_rtt'], ds[next_ts]['rtt']]

            # generate FUTURE_CHUNKS rows
            for i in range(Model.FUTURE_CHUNKS):
                row_i = row.copy()

                ts = next_ts + i * VIDEO_DURATION
                if ts in ds and 'trans_time' in ds[ts]:
                    row_i += [ds[ts]['size']]

                    assert(len(row_i) == Model.DIM_IN)
                    ret[i]['in'].append(row_i)
                    ret[i]['out'].append(ds[ts]['trans_time'])
    # print the first 1 record of the dictionary
    # print(f'{len(ret)} total number of records preprae input output')
    # print(len(ret[0]['in']))
    # print(ret[0]['out'])
    return ret


# Samples data for continual learning (CL) from the last CL_MAX_DAYS days. It applies a discount factor (CL_DISCOUNT) to prioritize recent data and calls prepare_input_output() to structure the data.
# Used in the continual learning process, this function samples and prepares the training data based on time windows. It helps the model train using more recent data without forgetting past trends.
def cl_sample(args, time_start, time_end, max_size, ret):
    raw_data = prepare_raw_data(args.yaml_settings,
                                time_start, time_end, args.cc)
    if not raw_data:
        # failed to sample valid data
        return 0

    raw_in_out = prepare_input_output(raw_data)

    ret_sample_size = None
    for i in range(Model.FUTURE_CHUNKS):
        real_size = len(raw_in_out[i]['in'])
        assert(real_size == len(raw_in_out[i]['out']))
        perm_indices = np.random.permutation(real_size)[:max_size]

        if ret_sample_size is None or len(perm_indices) < ret_sample_size:
            ret_sample_size = len(perm_indices)

        for j in perm_indices:
            ret[i]['in'].append(raw_in_out[i]['in'][j])
            ret[i]['out'].append(raw_in_out[i]['out'][j])

    return ret_sample_size


# Prepares data specifically for continual learning by invoking cl_sample() across multiple time windows, from recent days to older ones. The sampling size is adjusted by day using weights.
# This function is called when --cl (continual learning) is enabled. It handles the logic for gathering training data across multiple days to ensure continual learning is based on recent trends.
def prepare_cl_data(args):
    # calculate sampling weights and max data size to sample
    total_weights = 0
    for day in range(CL_MAX_DAYS):
        total_weights += CL_DISCOUNT ** day

    max_data_size = []
    for day in range(CL_MAX_DAYS):
        max_data_size.append(
            int((CL_DISCOUNT ** day / total_weights) * CL_MAX_DATA_SIZE))

    # training data set to return
    ret = [{'in':[], 'out':[]} for _ in range(Model.FUTURE_CHUNKS)]

    time_str = '%Y-%m-%dT%H:%M:%SZ'
    td = datetime.utcnow()
    today = datetime(td.year, td.month, td.day, td.hour, 0)

    # sample data from the past week
    for day in range(CL_MAX_DAYS):
        end_ts = today - timedelta(days=day)
        start_ts = today - timedelta(days=day+1)

        end_ts_str = end_ts.strftime(time_str)
        start_ts_str = start_ts.strftime(time_str)

        # sample data between 'day+1' and 'day' days ago, and save into 'ret'
        max_size = max_data_size[day]
        sample_size = cl_sample(args, start_ts_str, end_ts_str, max_size, ret)

        sys.stderr.write('Sampled {} data vs required {} data in day -{}\n'
                         .format(sample_size, max_size, day + 1))
        gc.collect()

    return ret


# Prints the distribution of output labels (transmission times in bins) and shows the single-label accuracy.
def print_stats(i, output_data):
    # print label distribution
    bin_sizes = np.zeros(Model.BIN_MAX + 1, dtype=int)
    for bin_id in output_data:
        bin_sizes[bin_id] += 1
    sys.stderr.write('[{}] label distribution:\n\t'.format(i))
    for bin_size in bin_sizes:
        sys.stderr.write(' {}'.format(bin_size))
    sys.stderr.write('\n')

    # predict a single label
    sys.stderr.write('[{}] single label accuracy: {:.2f}%\n'
                     .format(i, 100 * np.max(bin_sizes) / len(output_data)))


# Plots training and validation loss over epochs and saves the graph to a file.
def plot_loss(losses, figure_path):
    fig, ax = plt.subplots()

    if 'train' in losses:
        ax.plot(losses['train'], 'g--', label='training')
    if 'validate' in losses:
        ax.plot(losses['validate'], 'r-', label='validation')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid()
    ax.legend()

    fig.savefig(figure_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    sys.stderr.write('Saved plot to {}\n'.format(figure_path))


# Trains the model by processing the input data in batches, calculating the loss, and updating model weights using backpropagation. It handles training and validation if TUNING is enabled.
# Core function for training the neural network model. It's called within train_or_eval_model() to handle the actual training loop, update the model, and save checkpoints.
def train(i, args, model, input_data, output_data):
    if TUNING:
        # permutate input and output data before splitting
        perm_indices = np.random.permutation(len(input_data))
        input_data = input_data[perm_indices]
        output_data = output_data[perm_indices]

        # split training data into training/validation
        num_training = int(0.8 * len(input_data))
        train_input = input_data[:num_training]
        train_output = output_data[:num_training]
        validate_input = input_data[num_training:]
        validate_output = output_data[num_training:]
        sys.stderr.write('[{}] training set size: {}\n'
                         .format(i, len(train_input)))
        sys.stderr.write('[{}] validation set size: {}\n'
                         .format(i, len(validate_input)))

        validate_losses = []
    else:
        num_training = len(input_data)
        sys.stderr.write('[{}] training set size: {}\n'
                         .format(i, num_training))

    train_losses = []

    # number of batches
    num_batches = int(np.ceil(num_training / BATCH_SIZE))
    sys.stderr.write('[{}] total epochs: {}\n'.format(i, NUM_EPOCHS))

    # csv_file = f'transmission-values-{i}.csv'
    # trans_val_df = pd.read_csv(csv_file)

    predictions_list = []  # stores predictions

    # loop over the entire dataset multiple times
    for epoch_id in range(1, 1 + NUM_EPOCHS):
        # permutate data in each epoch
        perm_indices = np.random.permutation(num_training)

        running_loss = 0
        for batch_id in range(num_batches):
            start = batch_id * BATCH_SIZE
            end = min(start + BATCH_SIZE, num_training)
            batch_indices = perm_indices[start:end]

            # get a batch of input data
            batch_input = input_data[batch_indices]
            batch_output = output_data[batch_indices]

            # get predictions for the batch
            model.set_model_eval()
            predictions = model.predict(batch_input)
            predictions_list.append(predictions)
            model.set_model_train()

            # trans_val_df.loc[batch_indices, 'Predicted'] = predictions

            running_loss += model.train_step(batch_input, batch_output)
        running_loss /= num_batches

        # trans_val_df.to_csv(csv_file, index=False)

        # print info
        if TUNING:
            train_loss = model.compute_loss(train_input, train_output)
            validate_loss = model.compute_loss(validate_input, validate_output)
            train_losses.append(train_loss)
            validate_losses.append(validate_loss)

            train_accuracy = 100 * model.compute_accuracy(
                    train_input, train_output)
            validate_accuracy = 100 * model.compute_accuracy(
                    validate_input, validate_output)

            sys.stderr.write('[{}] epoch {}:\n'
                             '\ttraining: loss {:.3f}, accuracy {:.2f}%\n'
                             '\tvalidation: loss {:.3f}, accuracy {:.2f}%\n'
                             .format(i, epoch_id,
                                     train_loss, train_accuracy,
                                     validate_loss, validate_accuracy))
        else:
            train_losses.append(running_loss)
            sys.stderr.write('[{}] epoch {}: training loss {:.3f}\n'
                             .format(i, epoch_id, running_loss))
   
        # save checkpoints or the final model
        if epoch_id % CHECKPOINT == 0 or epoch_id == NUM_EPOCHS:
            if epoch_id == NUM_EPOCHS:
                suffix = ''
            else:
                suffix = '-checkpoint-{}'.format(epoch_id)

            model_path = path.join(args.save_model,
                                   'py-{}{}.pt'.format(i, suffix))
            model.save(model_path)
            sys.stderr.write('[{}] Saved model for Python to {}\n'
                             .format(i, model_path))

            model_path = path.join(args.save_model,
                                   'cpp-{}{}.pt'.format(i, suffix))
            meta_path = path.join(args.save_model,
                                  'cpp-meta-{}{}.json'.format(i, suffix))
            model.save_cpp_model(model_path, meta_path)
            sys.stderr.write('[{}] Saved model for C++ to {} and {}\n'
                             .format(i, model_path, meta_path))

            # plot losses
            losses = {}
            losses['train'] = train_losses
            if TUNING:
                losses['validate'] = validate_losses

            loss_path = path.join(args.save_model,
                                  'loss{}{}.png'.format(i, suffix))
            plot_loss(losses, loss_path)


# Handles both training and evaluation of the model. It creates or loads a model, normalizes input data, and either trains the model or evaluates its performance based on the provided arguments.
# It is called for each future chunk (model) during training or inference. It is a wrapper that calls training or evaluation depending on the command-line flags. This function also prints stats and saves models.
def train_or_eval_model(i, args, raw_in_data, raw_out_data):
    # does not seem to benefit from intra-op parallelism
    # print(raw_out_data)
    torch.set_num_threads(1)

    # create or load a model
    model = Model()
    if args.load_model:
        model_path = path.join(args.load_model, 'py-{}.pt'.format(i))
        model.load(model_path)
        sys.stderr.write('[{}] Loaded model from {}\n'.format(i, model_path))
    else:
        sys.stderr.write('[{}] Created a new model\n'.format(i))

    # normalize input data
    if args.inference:
        input_data = model.normalize_input(raw_in_data, update_obs=False)
    else:
        input_data = model.normalize_input(raw_in_data, update_obs=True)

    # discretize output data
    output_data = model.discretize_output(raw_out_data)

    # print some stats
    print_stats(i, output_data)

    if args.inference:
        model.set_model_eval()

        
        # sys.stderr.write('[{}] loss: {:.3f}, accuracy: {:.2f}%\n'
        #     .format(i, model.compute_loss(input_data, output_data),
        #             100 * model.compute_accuracy(input_data, output_data)))
        # # print input_data and output_data
        
        # sys.stderr.write('[{}] input_data: {}\n'.format(i, input_data))
        # sys.stderr.write('[{}] output_data: {}\n'.format(i, output_data))

        # # print len input data
        # sys.stderr.write('[{}] len input_data: {}\n'.format(i, len(input_data)))
        # sys.stderr.write('[{}] len each record input : {}\n'.format(i, len(input_data[0])))
        # sys.stderr.write('[{}] len output_data: {}\n'.format(i, len(output_data)))
        # sys.stderr.write('[{}] len each record output : {}\n'.format(i, len(output_data)))
        if i == 0:
            sys.stderr.write('[{}] test set size: {}\n'.format(i, len(input_data)))
            predictions = model.predict(input_data)


            # Display or return the predictions
            sys.stderr.write('[{}] raw output data:\n'.format(i))
            sys.stderr.write('{}\n'.format(raw_out_data))
            sys.stderr.write('[{}] predictions:\n'.format(i))
            sys.stderr.write('{}\n'.format(predictions))
            mse = calculate_mean_squared_error(predictions, raw_out_data)
            sys.stderr.write('[{}] Mean squared error:\n'.format(i))
            sys.stderr.write('{}\n'.format(mse))
    else:  # training
        
        model.set_model_train()

        # train a neural network with data
        train(i, args, model, input_data, output_data)


def calculate_mean_squared_error(predictions, actual):
    sum = 0
    print("Length of predictions: ", len(predictions))
    print("Length of actual: ", len(actual))
    for i in range(len(predictions)):
        sum += (predictions[i] - actual[i]) ** 2
    mse = sum / len(predictions)
    # print(f'Mean Squared Error: {mse}')
    # dump the predictions and actual values to a csv file 
    with open('predictions.csv', 'w') as f:
        for i in range(len(predictions)):
            f.write(f'{predictions[i]}, {actual[i]}\n')
    with open ('mse.txt', 'w') as f:
        f.write(f'Mean Squared Error: {mse}')
    return mse


# Orchestrates data preparation, validation, and training/inference. It creates parallel processes for training/evaluating models on multiple future chunks.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_settings')
    parser.add_argument('--from', dest='time_start',
                        help='datetime in UTC conforming to RFC3339')
    parser.add_argument('--to', dest='time_end',
                        help='datetime in UTC conforming to RFC3339')
    parser.add_argument('--cc', help='filter input data by congestion control')
    parser.add_argument('--load-model',
        help='folder to load {:d} models from'.format(Model.FUTURE_CHUNKS))
    parser.add_argument('--save-model',
        help='folder to save {:d} models to'.format(Model.FUTURE_CHUNKS))
    parser.add_argument('--enable-gpu', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--cl', action='store_true', help='continual learning')
    args = parser.parse_args()

    # validate and process args
    check_args(args)

    if not args.cl:
        # query InfluxDB and retrieve raw data
        raw_data = prepare_raw_data(args.yaml_settings,
                                    args.time_start, args.time_end, args.cc)
        # collect input and output data from raw data
        raw_in_out = prepare_input_output(raw_data)
    else:
        # continual learning
        raw_in_out = prepare_cl_data(args)

    gc.collect()

    # train or test FUTURE_CHUNKS models
    proc_list = []
    for i in range(Model.FUTURE_CHUNKS):
        proc = Process(target=train_or_eval_model,
                       args=(i, args,
                             raw_in_out[i]['in'], raw_in_out[i]['out'],))
        # trans_val_df = pd.DataFrame({
        #         'Actual': raw_in_out[i]['out'] ,
        #         'Predicted': [None] * len(raw_in_out[i]['out']) # placeholder for predicted values
        #     })
        # trans_val_df.to_csv(f'transmission-values-{i}.csv', index=False)
        
        proc.start()
        proc_list.append(proc)

    # wait for all processes to finish
    for proc in proc_list:
        proc.join()


if __name__ == '__main__':
    main()

### Run the code with this command for inferring the predictions
# python3 ttp.py ../settings.yml --load-model bbr-20220930-1/ --inference