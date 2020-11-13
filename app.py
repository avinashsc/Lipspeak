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
import shutil
import json

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

from config import load_args
configdl = load_args()
graph_dict = {
              'train': TransformerTrainGraph,
              'infer': TransformerInferenceGraph,
              }

# https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
#@app.route('/predict', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def handle_form():
    print("Lipspeak: Posted file: {}".format(request.files['file']))
    file = request.files['file']
    file.save('media/lipspeak/raw_videos/demo.mp4')
    print("Lipspeak: File saved at: 'media/lipspeak/raw_videos/demo.mp4'")
    print("Lipspeak: Resizing video")
    video_alignment_resizing()
    print("Lipspeak: Extracting features")
    evaluate_model()
    print("Lipspeak: Predict using KWS")
    evaluation(config)
    #encoded_img = get_response_image("data/demo/demo.png")
    #response =  { 'Status' : 'Success', 'ImageBytes': encoded_img}
    return jsonify({'index': 1})

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def collate_fn(batch):
  if True:
    return batch
  return default_collate(batch)

def calc_eer(scores, labels):
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    fnr = fnr*100
    fpr = fpr*100
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    if fpr[idxE] > fnr[idxE]:
        return fpr[idxE]
    else:
        return fnr[idxE]

def mean_average_precision_score(labels,scores,original_labels):
    total_positives = np.sum(original_labels)
    Iscores = np.argsort(-scores)
    labels_sorted = labels[Iscores]
    average_precision_array = []
    counter = 0
    for idx,val in enumerate(labels_sorted):
        if val ==1:
            counter +=1
            average_precision_array.append(counter/float(idx+1))
    mean_average_precision = np.sum(np.asarray(average_precision_array))/float(total_positives)
    return mean_average_precision

def recall_at_k(r, k, ground_truth):
    assert k >= 1
    r_2 = np.asarray(r)[:k] != 0
    if r_2.size != k:
        raise ValueError('Relevance score length < k')

    return np.sum(r_2)/float(np.sum(ground_truth))

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
        #print("DEBUG: batchword", batchword)
        #print("DEBUG: lstV_widx_sent", lstV_widx_sent)
        for k in range(0,batch_size):
            # lstV_widx_sent[k][0] is the video numpry array
            #print(f"[DEBUG] The type of lstV_widx_sent[k][0]: {str(type(lstV_widx_sent[k][0]))}")
            #print(f"[DEBUG] The length of lstV_widx_sent[k][0]: {len(lstV_widx_sent[k][0])}")
            #print(f"[DEBUG] lstV_widx_sent[k][0].size(0): {lstV_widx_sent[k][0].size(0)}")            
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

    if logger is None:
        logger = config.get_logger('test')

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
    
    test_dataset = DatasetV(num_words, num_phoneme_thr, cmu_dict_path, vis_feat_dir,split, data_struct_path, p_field_path, g_field_path, False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers =
        num_workers, pin_memory=pin_memory, shuffle=shuffle, drop_last=drop_last, collate_fn = collate_fn )
    
    Words = []
    for i, lstVwidx in enumerate(test_loader):
      for b in range(0, len(lstVwidx)):
        for w in lstVwidx[b][1]:
          if w != -1:
            Words.append(w)

    Words = np.unique(np.asarray(Words).astype('int32')).tolist()
    end = time.time()
    labels = []
    scores = []
    original_labels = []
    names = []

    #####EH Add code block for testing and multiple videos and saving results#####
    CMUwords = get_CMU_words(cmu_dict_path)
    result_dir = os.path.join(os.path.dirname(data_struct_path), "results")
    # print("[DEBUG]", result_dir)
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir, ignore_errors = False)
    os.mkdir(result_dir)
    
    video_names = get_video_names(data_struct_path, split)
    
    #with open(os.path.join(os.path.dirname(data_struct_path), "video_lengths.dict")) as f:
    #    video_lengths = json.load(f)
    results = {}
    
    #####EH End of the added code block Erik Hou#####

    for j, batchword in enumerate(Words):
      for i, lstVwidx in enumerate(test_loader):
        #####EH Add code block for testing and multiple videos and saving results#####
        result_sub_dir = os.path.join(result_dir, f"{video_names[i]}")
        os.makedirs(result_sub_dir, exist_ok = True)
        #####EH End of the added code block Erik Hou#####
        input, lens, target, vnames, view, start_times, end_times, rec_field, Ilens = transform_batch_test(lstVwidx, batchword, config)
        names = np.concatenate((names,vnames), axis=0)
        batch_size = input.shape[0]
        widx = np.asarray([batchword]*batch_size).astype('int32')
        target = torch.from_numpy(target).cuda(non_blocking=True)
        input = torch.from_numpy(input).float().cuda(non_blocking=True)
        widx = torch.from_numpy(widx).cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target.view(-1,1)).float()
        grapheme = []
        phoneme = []
        for w in widx:
          grapheme.append(test_dataset.get_GP(w)[0])
          phoneme.append(test_dataset.get_GP(w)[1])

        graphemeTensor = Variable(test_dataset.grapheme2tensor(grapheme)).cuda()
        phonemeTensor = Variable(test_dataset.phoneme2tensor(phoneme)).cuda()

        for k in range(0,len(target)):
            logits = []
            padding = math.ceil((rec_field-1)/2)
            input_loc =  input[k,:,:].unsqueeze(0).cpu().detach().numpy()
            input_loc = np.pad(input_loc, ((0,0), (padding,padding), (0,0)), 'constant', constant_values=(0, 0))
            for m in range(0,lens[k]):
              input_chunck = torch.from_numpy(input_loc).float().cuda(non_blocking=True)[:, 11-11+m:11+12+m, :]
              input_var_chunck= Variable(input_chunck)
              lens_loc = np.asarray([23])
              preds_loc = model(vis_feat_lens=lens_loc,p_lengths=None, phonemes=phonemeTensor[:,k].unsqueeze(1).detach(),
              graphemes=graphemeTensor[:-1][:,k].unsqueeze(1).detach(), vis_feats=input_var_chunck, use_BE_localiser=use_BE_localiser, epoch=74, config=config)
              logits.append(preds_loc["o_logits"][0][1][0].item())    
            logits = sigmoid(np.array(logits))
            plt.figure(figsize=(10,3))
            plt.xlabel('Time frames')
            plt.ylabel('Detection probability')
            plt.axvline(start_times[k],color='green')
            plt.axvline(end_times[k],color='green')
            plot1, = plt.plot(logits,'tab:blue',linewidth=3, color='blue')
            plt.tight_layout()

            #####EH Add code block for testing and multiple videos and saving results#####            
            #results[" ".join([CMUwords[batchword], video_names[i], str(video_lengths[video_names[i]])])] = logits
            results[" ".join([CMUwords[batchword], video_names[i]])] = logits
            file_path = os.path.join(result_sub_dir, f"{CMUwords[batchword]}.png")
            plt.savefig(file_path)
            plt.close()
            print("Demo output file created at", file_path)
            ######EH 
            #plt.savefig("data/lipspeak/demo.png")
            #plt.clf()
    #print("Demo output file created at data/lipspeak/demo.png")

def video_alignment_resizing():    
    output_video_dir = "media/lipspeak"
    raw_video_dir = os.path.join(output_video_dir, "raw_videos")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    for video_name in os.listdir(raw_video_dir):
        #print("DEBUG: video_name is", video_name)
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
  
            frame = Image.fromarray(frame)
            frame = frame.rotate(-90)
            frame = np.asarray(frame)
            try:
                processed_frame = frame[62:640, 343:921]
                processed_frame = cv2.resize(processed_frame, (160, 160))
                out.write(processed_frame)
            except Exception as e:
                print("DEBUG: resize error", str(e))

        print(f"Saved file at: {output_video_path}")
        out.release()

    cv2.destroyAllWindows

def evaluate_model():
    np.random.seed(configdl.seed)
    tf.set_random_seed(configdl.seed)

    val_g, val_epoch_size, chars, sess, val_gen = init_models_and_data(istrain=0)

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
    
    Loss = []
    Cer = []
    Wer = []
    
    progbar = Progbar(target=n_batches, verbose=1, stateful_metrics=['t'])
    print ('Strating validation Loop')
  
    feature_dir = configdl.feat_output_dir
    v_names = write_query_file(feature_dir)
    feature_dir = os.path.join(feature_dir, "features")
    os.makedirs(feature_dir, exist_ok = False)
    
    video_frame_count = {}
    #####EH End of the added code block Erik Hou#####
    
    for i in range(n_batches):
    
        ###EH add code to extract video length in terms of number of frames###
        #x, y, vid_lens =  val_gen.next()
        ###EH End of code added by Erik to extract video length###
        ### original statement ###
        x, y =  val_gen.next()
        #print(f"[DEBUG] The type of x: {str(type(x))}")
        #print(f"[DEBUG] The meta of x: {x.shape}")
        #print(f"[DEBUG] The type of y: {str(type(y))}")
        # print(f"[DEBUG] The meta of y: {y.shape}")
        if len(x) == 1: x = x[0]
        if len(y) == 1: y = y[0]
    
        #print(f"[DEBUG] The type of x: {str(type(x))}")
        #print(f"[DEBUG] The meta of x: {x.shape}")
        #print(f"[DEBUG] The type of y: {str(type(y))}")
        #print(f"[DEBUG] The meta of y: {y.shape}")
        
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
            #original_video_len = vid_lens[0]
            #print(f"[DEBUG] The type of features_to_extract: {str(type(features_to_extract))}")
            #print(f"[DEBUG] The meta of features_to_extract: {features_to_extract.shape}")
            #print(f"[DEBUG] The original video length: {original_video_len}")
            
            file_path = os.path.join(feature_dir, f"{v_names[i]}.npy")
            with open(file_path, 'wb') as f:
                np.save(f, features_to_extract)
            print("feature output file created at", file_path)
      
            #####EH End of the added code block Erik Hou#####
        ###EH End of code added by Erik to extract video length###
  
def init_models_and_data(istrain):
    print ('Loading data generators')
    val_gen, val_epoch_size = setup_generators()
    #print ('Done')
    #print(f"[DEBUG] The content of config.gpu_id: {configdl.gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(configdl.gpu_id)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=sess_config)
    #print(f"[DEBUG] The content of config.lm_path: {configdl.lm_path}")
    if configdl.lm_path:
        # initialize singleton rnn so that RNN tf graph is created first
        beam_batch_size = 1
        lm_handle = CharRnnLmWrapperSingleton(batch_size=beam_batch_size,
                                          sess=sess,
                                          checkpoint_path=configdl.lm_path)
  
    #print(f"[DEBUG] The content of config.graph_type: {configdl.graph_type}")
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

def rel_edist(tr, pred):
    return editdistance.eval(tr,pred) / float(len(tr))

def get_CMU_words(CMU_path):
    words = []
    phonemes = []
    with open(CMU_path) as f:
        lines = f.readlines()
    for wcnt, line in enumerate(lines):
        grapheme, phoneme = line.split(" ",1)
        words.append(grapheme)
        phonemes.append(phoneme.split(" "))
    return words

def get_video_names(data_struct_path, split):
    with open(data_struct_path) as f:
        test_cases = json.load(f)[split]
    return [test_case["fn"] for test_case in test_cases]

def write_query_file(feature_dir):
    if os.path.isdir(feature_dir):
        shutil.rmtree(feature_dir, ignore_errors = False)
    os.makedirs(feature_dir, exist_ok = True)
  
    CMUwords = get_CMU_words(configdl.dict_file)
  
    #print(f"[DEBUG] config.query_type: {configdl.query_type}")
    
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
            # print(v_names[0][:-4])
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
    app.run(host='0.0.0.0')
