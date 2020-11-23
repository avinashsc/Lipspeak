import argparse
import time
import torch
import random
from tqdm import tqdm
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
import data_loader.datasets as module_data
import model.loss as module_loss
import model.metric as module_met
import model.model as module_arch
from torch.utils.data import DataLoader
from utils.util import canonical_state_dict_keys
from parse_config import ConfigParser
from model.metric import AverageMeter
from data_loader.datasets import DatasetV
from torch.utils.data.dataloader import default_collate
import sys
import json
import glob
from scipy.special import expit as sigmoid
from sklearn.metrics import average_precision_score
import pkg_resources
pkg_resources.require("matplotlib==3.2.0rc1")
import matplotlib.pyplot as plt
sys.setrecursionlimit(1500)

from flask import Flask, request, jsonify
from PIL import Image
import io
from base64 import encodebytes
import cv2
import editdistance
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from datas.list_generator import ListGenerator
from language_model.char_rnn_lm import CharRnnLmWrapperSingleton
from lip_model.training_graph import TransformerTrainGraph
from lip_model.inference_graph import TransformerInferenceGraph
import json
import shutil

import threading
import copy
import queue

app = Flask(__name__)
args = argparse.ArgumentParser()
config = ConfigParser(args)
model = config.init('arch', module_arch)
logger = config.get_logger('test')
tic = time.time()
with open(os.path.join('./misc/pretrained_models', 'KWS_Net.pth'), 'rb') as f:
    checkpoint = torch.load(f)

state_dict = canonical_state_dict_keys(checkpoint['state_dict'])
model.load_state_dict(state_dict)   
logger.info(f"Finished loading ckpt in {time.time() - tic:.3f}s")

logger.info(f"CUDA device count: {torch.cuda.device_count()}")
device_count = torch.cuda.device_count()
models = []
for device_ind in range(device_count):
    device = f"cuda:{device_ind}"
    models.append(copy.deepcopy(model).to(device))
    models[device_ind].eval()
    
from config import load_args
configdl = load_args()
graph_dict = {
              'train': TransformerTrainGraph,
              'infer': TransformerInferenceGraph,
              }

def init_models_and_data(istrain):

    print ('Loading data generators')
    val_gen, val_epoch_size = setup_generators()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(configdl.gpu_id)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=sess_config)

    if configdl.lm_path:
        # initialize singleton rnn so that RNN tf graph is created first
        beam_batch_size = 1
        lm_handle = CharRnnLmWrapperSingleton(batch_size=beam_batch_size,
                                          sess=sess,
                                          checkpoint_path=configdl.lm_path)

    TransformerGraphClass = graph_dict[configdl.graph_type]

    (shapes_in, dtypes_in), (shapes_out, dtypes_out) = \
                                                       TransformerGraphClass.get_model_input_target_shapes_and_types()

    go_idx = val_gen.label_vectorizer.char_indices[val_gen.label_vectorizer.go_token]
    x_val = tf.placeholder(dtypes_in[0], shape=shapes_in[0])
    prev_shape = list(shapes_out[0])
    if configdl.test_aug_times : prev_shape[0] *= configdl.test_aug_times
    prev_ph = tf.placeholder(dtypes_out[0], shape=prev_shape)
    y_ph = tf.placeholder(dtypes_out[0], shape=shapes_out[0])
    y_val = [prev_ph, y_ph]

    chars = val_gen.label_vectorizer.chars
    val_g = TransformerGraphClass(x_val,
                                y_val,
                                is_training=False,
                                reuse=tf.AUTO_REUSE,
                                go_token_index=go_idx,
                                chars=chars)
    print("Validation Graph loaded")

    sess.run(tf.tables_initializer())

    load_checkpoints(sess)

    return val_g, val_epoch_size, chars, sess, val_gen

def load_checkpoints(sess, var_scopes = ('encoder', 'decoder', 'dense')):
    checkpoint_path =  configdl.lip_model_path
    if checkpoint_path:
        if os.path.isdir(checkpoint_path):
            checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        else:
            checkpoint = checkpoint_path

    if configdl.featurizer:

        if checkpoint_path:
            from tensorflow.contrib.framework.python.framework import checkpoint_utils
            var_list = checkpoint_utils.list_variables(checkpoint)
            for var in var_list:
                if 'visual_frontend' in var[0]:
                    var_scopes = var_scopes + ('visual_frontend',)
                    break

        if not 'visual_frontend' in var_scopes:
            featurizer_vars = tf.global_variables(scope='visual_frontend')
            featurizer_ckpt = tf.train.get_checkpoint_state(configdl.featurizer_model_path)
            featurizer_vars = [var for var in featurizer_vars if not 'Adam' in var.name]
            tf.train.Saver(featurizer_vars).restore(sess, featurizer_ckpt.model_checkpoint_path)

    all_variables = []
    for scope in var_scopes:
        all_variables += [var for var in tf.global_variables(scope=scope)
                          if not 'Adam' in var.name ]
    if checkpoint_path:
        tf.train.Saver(all_variables).restore(sess, checkpoint)

    print("Restored saved model {}!".format(checkpoint))

def setup_generators(verbose=False):
    val_gen = ListGenerator(data_list=configdl.data_list)
    val_epoch_size = val_gen.calc_nbatches_per_epoch()
    return val_gen, val_epoch_size

tic = time.time()
np.random.seed(configdl.seed)
tf.set_random_seed(configdl.seed)
val_g, val_epoch_size, chars, sess, val_gen = init_models_and_data(istrain=0)
logger.info(f"Finished initializing Deep Lip Reading model in {time.time() - tic:.3f}s")

# https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("Lipspeak: Posted file: {}".format(request.files['file']))
    tic = time.time()
    file = request.files['file']
    file.save('media/lipspeak/raw_videos/demo.mp4')
    #print("Lipspeak: File saved at: 'media/lipspeak/raw_videos/demo.mp4'")
    logger.info(f"Finished saving raw video in {time.time() - tic:.3f}s")

    #https://stackoverflow.com/questions/47679227/using-python-to-send-json-and-files-to-flask
    tic = time.time()
    print("Processing Phrasebook")
    queries = json.loads(request.form['phrasebook'])

    with open('data/vocab/lipspeak/testphrases.json') as json_file:
        phrases = json.load(json_file)

    testdict = {}
    for key, value in queries.items():
        for x in value:
            x = x.strip()
            testdict[x] = phrases[x]
            with open('data/vocab/lipspeak/test_phrases.json', 'w') as fp:
                json.dump(testdict, fp)

    CMUwords1, phonemes1 = get_CMU_words("data/vocab/cmudict.dict")
    
    with open('data/vocab/lipspeak/test_phrases.json') as f:
        test_cases = json.load(f)

    dict_lines = []

    for full_phrase in test_cases:
        for phrase_type in test_cases[full_phrase]:
            for test_phrase in test_cases[full_phrase][phrase_type]:
                test_phonemes = []
                for word in test_phrase.split(" "):
                    test_phonemes.append(" ".join(phonemes1[CMUwords1.index(word)]).replace("\n",""))
                dict_lines.append(test_phrase.replace("_", "") + " " + " ".join(test_phonemes) + "\n")

    with open("data/vocab/lipspeak/testdict.dict", "w") as f:
        f.writelines(dict_lines)
    logger.info(f"Finished processing phrasebook in {time.time() - tic:.3f}s")
  
    tic = time.time()
    print("Lipspeak: Resizing video")
    video_alignment_resizing()
    logger.info(f"Finished resizing video in {time.time() - tic:.3f}s")
    tic = time.time()
    print("Lipspeak: setup generator for deep lip read")
    val_gen, val_epoch_size = setup_generators()
    logger.info(f"Finished setup generator in {time.time() - tic:.3f}s")
    print("Lipspeak: Extracting features")
    tic = time.time()
    evaluate_model(val_g, val_epoch_size, chars, sess, val_gen)
    logger.info(f"Finished extracting features in {time.time() - tic:.3f}s")
    print("Lipspeak: Predict using KWS")
    tic = time.time()    
    kws_prediction = evaluation(config)
    logger.info(f"Finished evaluating KWS model in {time.time() - tic:.3f}s")    

    return jsonify({'index': int(kws_prediction)})

def collate_fn(batch):
  if True:
    return batch
  return default_collate(batch)

def transform_batch_test(lstV_widx_sent, batchword, config):
        
        target = []
        lens = []
        vnames = []
        vidx = []
        view = []
        batch_size = len(lstV_widx_sent)
        start_times = []
        end_times = []
        lstV_widx_sent_real = []
        for k in range(0,batch_size):
            if lstV_widx_sent[k][0].size(0)>1:
                lstV_widx_sent_real.append(lstV_widx_sent[k])
        batch_size = len(lstV_widx_sent_real)
        for k in range(0,batch_size):
            lens.append(lstV_widx_sent_real[k][0].size(0))
            TN = 1 if any(x == batchword for x in lstV_widx_sent_real[k][1]) else 0
            target.append(TN)
            if TN == 0:
              start_times.append(0)
              end_times.append(0)
            else:
              for i, x in enumerate(lstV_widx_sent_real[k][1]):
                if x ==batchword:
                  start_times.append(lstV_widx_sent_real[k][4][i])
                  end_times.append(lstV_widx_sent_real[k][5][i])
            vnames.append(lstV_widx_sent_real[k][2])
            view.append(lstV_widx_sent_real[k][3])
        lens = np.asarray(lens)
        target = np.asarray(target)
        start_times = np.asarray(start_times)
        end_times=np.asarray(end_times)
        Ilens = np.argsort(-lens)
        lens = lens[Ilens]
        target = target[Ilens]
        start_times = start_times[Ilens]
        end_times = end_times[Ilens]
        vnames = [vnames[i] for i in Ilens]
        view = [view[i] for i in Ilens]
        max_len = lens[0]
        max_out_len,rec_field, offset = in2out_idx(max_len)
        batchV = np.zeros((batch_size,max_len,lstV_widx_sent_real[0][0].size(1))).astype('float')
        for i in range(0, batch_size):
            batchV[i,:lens[i],:] = lstV_widx_sent_real[Ilens[i]][0].clone()
        return batchV, lens, target, vnames, view, start_times, end_times, rec_field, Ilens


def in2out_idx(idx_in):
  layers = [
    { 'type': 'conv3d', 'n_channels': 32,  'kernel_size': (1,5,5), 'stride': (1,2,2), 'padding': (0,2,2)  ,
          'maxpool': {'kernel_size' : (1,2,2), 'stride': (1,2,2)} },

    { 'type': 'conv3d', 'n_channels': 64, 'kernel_size': (1,5,5), 'stride': (1,2,2), 'padding': (0,2,2),
       'maxpool': {'kernel_size' : (1,2,2), 'stride': (1,2,2)}
      },

  ]
  layer_names = None
  from misc.compute_receptive_field import calc_receptive_field
  idx_out, _, rec_field, offset = calc_receptive_field(layers, idx_in, layer_names)
  return idx_out, rec_field, offset


def evaluation(config, logger=None):

    def infer_batch(batch_output, widx_list, config, device_ind, queue, model):
        logger.info(f"CUDA device context {device_ind}")
        with torch.cuda.device(device_ind):
            input, lens, target, vnames, view, start_times, end_times, rec_field, Ilens = batch_output
            batch_size = input.shape[0]
            target = torch.from_numpy(target).cuda(non_blocking=True)
            input = torch.from_numpy(input).float().cuda(non_blocking=True)
            input_var = Variable(input)
            target_var = Variable(target.view(-1,1)).float()
            grapheme = []
            phoneme = []
            for w in widx_list:
                grapheme.append(test_dataset.get_GP(w)[0])
                phoneme.append(test_dataset.get_GP(w)[1])
                batchword_str = ''.join(grapheme[0])
                
            graphemeTensor = Variable(test_dataset.grapheme2tensor(grapheme)).cuda()
            phonemeTensor = Variable(test_dataset.phoneme2tensor(phoneme)).cuda()

            for w in range(len(widx_list)):
                for k in range(0,len(target)):
                    logits = []
                    padding = math.ceil((rec_field-1)/2)
                    input_loc =  input[k,:,:].unsqueeze(0).cpu().detach().numpy()
                    input_loc = np.pad(input_loc, ((0,0), (padding,padding), (0,0)), 'constant', constant_values=(0, 0))
                    for m in range(0,lens[k]):
                        input_chunck = torch.from_numpy(input_loc).float().cuda(non_blocking=True)[:, 11-11+m:11+12+m, :]
                        input_var_chunck= Variable(input_chunck)
                        lens_loc = np.asarray([23])
                        preds_loc = model(vis_feat_lens=lens_loc,p_lengths=None, phonemes=phonemeTensor[:,w].unsqueeze(1).detach(), graphemes=graphemeTensor[:-1][:,w].unsqueeze(1).detach(), vis_feats=input_var_chunck, use_BE_localiser=use_BE_localiser, epoch=74, config=config)
                        logits.append(preds_loc["o_logits"][0][1][0].item())
                    logits = sigmoid(np.array(logits))
                    queue.put((widx_list[w], logits))
                
    if logger is None:
        logger = config.get_logger('test')

    device_count = torch.cuda.device_count()
    queues = []
    for device_ind in range(device_count):
        queues.append(queue.Queue())

    num_words = config["dataset"]["args"]["num_words"] #135091
    num_phoneme_thr = config["dataset"]["args"]["num_phoneme_thr"]
    split = config["dataset"]["args"]["split"]
    cmu_dict_path = config["dataset"]["args"]["cmu_dict_path"]
    data_struct_path = config["dataset"]["args"]["data_struct_path"]
    p_field_path = config["dataset"]["args"]["field_vocab_paths"]["phonemes"]
    g_field_path = config["dataset"]["args"]["field_vocab_paths"]["graphemes"]
    vis_feat_dir = config["dataset"]["args"]["vis_feat_dir"]
    batch_size = config["data_loader"]["args"]["batch_size"]
    shuffle = config["data_loader"]["args"]["shuffle"] 
    drop_last = config["data_loader"]["args"]["drop_last"]
    pin_memory = config["data_loader"]["args"]["pin_memory"]
    num_workers = config["data_loader"]["args"]["num_workers"]
    g2p =  config["arch"]["args"]["g2p"]
    use_BE_localiser = config["arch"]["args"]["rnn2"]

    #tic = time.time()
    test_dataset = DatasetV(num_words, num_phoneme_thr, cmu_dict_path, vis_feat_dir,split, data_struct_path, p_field_path, g_field_path, False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers =
        num_workers, pin_memory=pin_memory, shuffle=shuffle, drop_last=drop_last, collate_fn = collate_fn )

    #logger.info(f"Finished dataset loading in {time.time() - tic:.3f}s")
    #tic = time.time()
    
    Words = []
    for i, lstVwidx in enumerate(test_loader):
      for b in range(0, len(lstVwidx)):
        for w in lstVwidx[b][1]:
          if w != -1:
            Words.append(w)

    Words = np.unique(np.asarray(Words).astype('int32')).tolist()
    #end = time.time()
    labels = []
    scores = []
    original_labels = []
    names = []

    query_words,_ = get_CMU_words(cmu_dict_path)
    results = [None] * len(Words)

    logger.info(f"Start Inference:")
    #tic = time.time()
    batchword = 0
    for i, lstVwidx in enumerate(test_loader):
    	batch_output = transform_batch_test(lstVwidx, batchword, config)
    
    word_intervals = [len(Words)//device_count] * device_count
    remainder = len(Words) % device_count
    for i in range(remainder):
    	word_intervals[i] += 1
    	
    start_pos = 0
    for i in range(len(word_intervals)):
    	end_pos = start_pos + word_intervals[i]
    	word_intervals[i] = (start_pos, end_pos)
    	start_pos = end_pos
    print(word_intervals)

    threads = []
    for i in range(len(word_intervals)):
        threads.append(threading.Thread(target=infer_batch, args=(batch_output, Words[word_intervals[i][0]:word_intervals[i][1]], config, i%device_count, queues[i%device_count], models[i%device_count],)))
    
    for i in range(len(word_intervals)):
        threads[i].start()
    
    for i in range(len(word_intervals)):
        threads[i].join()
    
    for i in range(len(word_intervals)):
        while not queues[i].empty():
            j, logits = queues[i].get()
            results[j] = logits

    #logger.info(f"Finished Inference: time spent {round(time.time()-tic,3)}")

    #####EH inferece_code_block#####
    prob_threshold = 0.3
    results = np.stack(results, axis=0)
    probs = np.sum((results-prob_threshold).clip(0,1), axis = 1)
    if np.sum(probs) == 0:
        pred = -1
    else:
        pred = np.argmax(probs)
    print(f"Prediction is {pred}, which is {'unknown' if pred == -1 else query_words[pred]}")
    #logger.info(f"Finished KWS in {time.time() - tic:.3f}s")
    return pred
    #####EH inferece_code_block#####    

def video_alignment_resizing():    
    global video_len
    output_video_dir = "media/lipspeak"
    raw_video_dir = os.path.join(output_video_dir, "raw_videos")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    for video_name in os.listdir(raw_video_dir):
        if video_name[-4:].lower() not in [".mp4", ".mov"]: continue
        input_video_path = os.path.join(raw_video_dir, video_name)
        
        video_stream = cv2.VideoCapture(input_video_path)

        output_video_path = os.path.join(output_video_dir, video_name[:video_name.find(".")]+".mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (160,160))
        
        while 1:
            still_reading, frame = video_stream.read()
            out.write(frame)
            if not still_reading:
                video_stream.release()
                break

            try:
                processed_frame = frame[210:572, 55:417]
                processed_frame = cv2.resize(processed_frame, (160, 160))
                out.write(processed_frame)
                video_len += 1
            except Exception as e:
                print("DEBUG: resize error", str(e))

        video_len = video_len if video_len % 2 == 0 else video_len + 1
        print(f"Saved file at: {output_video_path}")
        out.release()

    cv2.destroyAllWindows

def evaluate_model(val_g, val_epoch_size, chars, sess, val_gen):
    np.random.seed(configdl.seed)
    tf.set_random_seed(configdl.seed)

    tb_writer = None
    if configdl.tb_eval:
      import shutil
      try: shutil.rmtree('eval_tb_logs')
      except: pass
      tb_logdir = os.path.join(os.getcwd(), 'eval_tb_logs' , 'val')
      tb_writer = tf.summary.FileWriter(tb_logdir, sess.graph)

    with sess.as_default():
      for _ in range(configdl.n_eval_times):
        validation_loop(sess, val_g,
            val_epoch_size,
            chars = chars,
            val_gen = val_gen,
            tb_writer = tb_writer,
            )
    print("Done")

def validation_loop(sess, g, n_batches, chars=None, val_gen = None, tb_writer=None):
    global video_len
    print(f"Video length is {video_len}")

    Loss = []
    Cer = []
    Wer = []
    
    progbar = Progbar(target=n_batches, verbose=1, stateful_metrics=['t'])
    print ('Starting validation Loop')
  
    feature_dir = configdl.feat_output_dir
    v_names = write_query_file(feature_dir)
    feature_dir = os.path.join(feature_dir, "features")
    os.makedirs(feature_dir, exist_ok = False)
    
    video_frame_count = {}
    
    for i in range(n_batches):
        x, y =  val_gen.next()
        if len(x) == 1: x = x[0]
        if len(y) == 1: y = y[0]
        # -- Autoregressive inference
        preds = np.zeros((configdl.batch_size, configdl.maxlen), np.int32)

        tile_preds = configdl.test_aug_times
        # -- For train graph feed in the previous step's predictions manually for the next
        if not 'infer' in configdl.graph_type:
            prev_inp = np.tile(preds, [configdl.test_aug_times,1]) if tile_preds else preds
            feed_dict = {g.x: x, g.prev: prev_inp, g.y: y}
      
            #####EH Add code block for testing and multiple videos and saving results#####
            features_to_extract = sess.run( g.feats, feed_dict)
            features_to_extract = features_to_extract[0, :, :]
            middle_pos = features_to_extract.shape[0] // 2
            print(f"feature length before trimming: {features_to_extract.shape[0]}")
            features_to_extract = features_to_extract[middle_pos-video_len//2:middle_pos+video_len//2,:]
            print(f"feature length before trimming: {features_to_extract.shape[0]}")

            video_len = 0
            file_path = os.path.join(feature_dir, f"{v_names[i]}.npy")
            with open(file_path, 'wb') as f:
                np.save(f, features_to_extract)
            print("feature output file created at", file_path)
            #####EH End of the added code block Erik Hou#####
        ###EH End of code added by Erik to extract video length###

def get_CMU_words(CMU_path):
    words = []
    phonemes = []
    with open(CMU_path) as f:
        lines = f.readlines()
    for wcnt, line in enumerate(lines):
        grapheme, phoneme = line.split(" ",1)
        words.append(grapheme)
        phonemes.append(phoneme.split(" "))
    return words, phonemes

def get_video_names(data_struct_path, split):
    with open(data_struct_path) as f:
        test_cases = json.load(f)[split]
    return [test_case["fn"] for test_case in test_cases]

def write_query_file(feature_dir):
    if os.path.isdir(feature_dir):
        shutil.rmtree(feature_dir, ignore_errors = False)
    os.makedirs(feature_dir, exist_ok = True)
  
    CMUwords,_ = get_CMU_words(configdl.dict_file)
  
    if configdl.query_type == "word":
        with open(configdl.data_list) as f:
            lines = f.readlines()
      
        v_names = []
        Dsplitsdemo = {"test":[]}
      
        for line in lines:
            v_name, text = line.split(",", 1)
            v_names.append(v_name)
        
            test_words = text.strip().replace(",", "").replace("\n", "").lower().split(" ")
            test_word_indices = []
        
            print(text, test_words)
        
            for test_word in test_words:
                try:
                    test_word_ind = CMUwords.index(test_word)
                except ValueError:
                    test_word_ind = -1
                test_word_indices.append(test_word_ind)
        
            test_words_len = len(test_word_indices)
            Dsplitsdemo["test"].append({
                "end_word": [0] * test_words_len, 
                "start_word": [0] * test_words_len, 
                "widx": test_word_indices, 
                "fn": v_name[:-4]})
    
    elif configdl.query_type == "phrase":
        with open(configdl.data_list) as f:
            lines = f.readlines()
        v_names = []
        Dsplitsdemo = {"phrase":[]}
        for line in lines:
            v_name, text = line.split(",", 1)
            v_names.append(v_name)

        Dsplitsdemo["phrase"].append({
            "end_word": [0] * len(CMUwords), 
            "start_word": [0] * len(CMUwords), 
            "widx": [i for i in range(len(CMUwords))], 
            "fn": v_names[0][:-4]})
    
        for i in range(1, len(v_names)):
            Dsplitsdemo["phrase"].append({
                "end_word": [], 
                "start_word": [], 
                "widx": [], 
                "fn": v_names[i][:-4]})
    else:
        print("Error!!!!!, wrong query type")
        return
  
    file_path = os.path.join(feature_dir, "Dsplitsdemo.json")
    with open(file_path, 'w') as f:
        f.write(json.dumps(Dsplitsdemo))
  
    return v_names

if __name__ == '__main__':
    video_len = 0
    app.run(host='0.0.0.0')
