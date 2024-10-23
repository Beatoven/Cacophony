import jax, flax
import jax.numpy as jnp
import tensorflow as tf
from einops import rearrange
import scipy
from tqdm import tqdm
import soundfile as sf
import numpy as np
import json
import os
import gc

from src.caco.load_model import load_caco
from src.caco.dataset import Batch, DatasetConfig,  _dataset_process_map, _tokenize_and_numpy
from src.caco.caco_eval_utils import load_from_list

from dataset_processors import BeatovenProcessor

# eval 2: (ZS) text to audio retrieval on audiocaps test
#######################################
# In retrieval task: 
# 1) compute all text embedding
# 2) compute all audio embedding
# 3a) in text to audio: rank the top audio embeddings on the given text embedding
# 3b) in audio to text: rank the top text embeddings on the given audio embedding
#######################################

ckpt_path = "Cacophony.ckpt"
caco_model_dict = load_caco(ckpt_path, use_decoder=True)
caco_model = caco_model_dict['caco_model']

def compute_audio_embedding(audio_batch, model_params):
    return caco_model.apply(
        {'params': model_params},
        audio_patches=audio_batch['audio_patches'],
        audio_time_inds=audio_batch['audio_time_inds'],
        audio_freq_inds=audio_batch['audio_freq_inds'],
        audio_mask=audio_batch['audio_mask'],
        deterministic=True,
        return_hidden_state=False,
        normalize=True,
        method=caco_model.get_audio_embedding,
    )

def compute_text_embedding(text_batch, model_params):
    return caco_model.apply(
        {'params': model_params},
        text_input_ids=text_batch['text_input_ids'], 
        text_mask=text_batch['text_mask'],
        deterministic=True,
        return_hidden_state=False,
        normalize=True,
        method=caco_model.get_text_embedding,
    )

t_apply = jax.pmap(compute_text_embedding, axis_name='dp')
a_apply = jax.pmap(compute_audio_embedding,axis_name='dp')
caco_params = flax.jax_utils.replicate(caco_model_dict['caco_params'], devices=jax.local_devices())
tokenizer = caco_model_dict['tokenizer']

PyTreeDef = type(jax.tree_util.tree_structure(None))

def get_train_input(
    batch: Batch
) -> PyTreeDef:
    batch = dict(
        audio_patches=batch.audio_patches,
        audio_time_inds=batch.audio_time_inds,
        audio_freq_inds=batch.audio_freq_inds,
        audio_mask=batch.audio_mask,
        text_input_ids=batch.text_input_ids,
        text_mask=batch.text_mask,
    )
    batch = jax.tree_util.tree_map(
        lambda x: rearrange(jnp.asarray(x), '(d b) ... -> d b ...', d=jax.local_device_count()),
        batch
    )
    return batch

@tf.function
def load_audio(audio_path, dataset_sampling_rate):
    audiowav, _ = sf.read(audio_path)
    audiowav = audiowav.astype(np.float32)
    if len(audiowav.shape) > 1:
        audiowav = np.mean(audiowav, axis=-1)

    if dataset_sampling_rate != 16000:
        new_num_samples = round(audiowav.shape[-1]*float(16000)/dataset_sampling_rate)
        audiowav = scipy.signal.resample(audiowav, new_num_samples)

    return audiowav

def prepare_audio_batch(audiowav, audio_description, datasetconfig):

    data_dict = load_from_list(audiowav, audio_description)
    d_ = _dataset_process_map(data_dict, [0, 1], datasetconfig)
    d = {}
    for d_item in d_:
        d[d_item] = tf.expand_dims(d_[d_item], axis=0)
    d = _tokenize_and_numpy(d, datasetconfig, tokenizer)
    batch = get_train_input(d)

    return batch, data_dict

def audio_retrieval(dataprocessor, datasetconfig, eval_split=''):
    filepaths, descriptions, _ = dataprocessor.get_filepaths_and_descriptions(current_split=eval_split)

    dataset_len = len(filepaths)
    
    #all_text = []
    #all_text_embeddings = []
    all_audio = []
    all_audio_embeddings = []
    #gt_audio_text = {}
    #gt_text_audio = {}
    dump_idx = 0
    
    os.makedirs('rw_embeddings',exist_ok=True)

    for file_idx in tqdm(range(dataset_len)):

        if file_idx % 200 == 0:
            gc.collect()

        audio_name = filepaths[file_idx].split('/')[-1]
        #gt_audio_text[audio_name] = []

        # get text embeddings
        #audio_descriptions = descriptions[audio_name]['description']
        if not os.path.isfile('embeddings/' + audio_name + '.json'):
            print(audio_name)
            audiowav = load_audio(filepaths[file_idx], dataprocessor.config.sampling_rate)
            batch, data_dict = prepare_audio_batch(audiowav, '', datasetconfig)

            #text_embedding = t_apply(batch, caco_params)

            # prepare for text embedding
            #text_str = bytes.decode(data_dict['text'][0].numpy())
            #gt_audio_text[audio_name].append(text_str) 
            #gt_text_audio[text_str] = audio_name
            #all_text.append(text_str)
            
            #all_text_embeddings.append(text_embedding)

            # get audio embedding
            audio_embedding = a_apply(batch, caco_params)

            tmp_dict={audio_name:audio_embedding[0][0].tolist()}

            with open('rw_embeddings/' + audio_name + '.json', 'w') as fp:
                json.dump(tmp_dict, fp)


    #all_text_embeddings = jnp.concatenate(all_text_embeddings, axis=0)



    #logits_ar=jnp.squeeze(all_text_embeddings, axis=1) @ jnp.squeeze(all_audio_embeddings.T, axis=1)
    
    # evaluation: audio to text
    #at_indices = jnp.argsort(jnp.transpose(-logits_ar), axis=-1)
    #compute_retrieval_metric(at_indices, all_audio, all_text, gt_audio_text)

    # evaluation: text to audio
    #print('text to audio retrieval:')
    #ta_indices = jnp.argsort(-logits_ar, axis=-1)
    #compute_retrieval_metric(ta_indices, all_text, all_audio, gt_text_audio, 'ta')

if __name__ == '__main__':


    audio_seg_time = 16
    total_samples = 16000 * audio_seg_time
    max_patches = (total_samples * 8 // 160 // 16) 
    CommondataConfig = DatasetConfig(batch_size=1,
                                        patches_seq_len=max_patches,
                                        time_patch_size=16,
                                        freq_patch_size=16,
                                        max_text_len=100,
                                        synthetic_prob=0.8)

    beatovenprocessor = BeatovenProcessor()
    audio_retrieval(beatovenprocessor, CommondataConfig, 'evaluation')
