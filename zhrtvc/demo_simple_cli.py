import os
#禁用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from synthesizer import hparams
from synthesizer.utils import audio
from encoder import inference as encoder
# from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import time

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="../models/encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path,
                        default="../models/synthesizer/saved_models/logs-syne/",  # pretrained
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="../models/vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("-o", "--wav_output_dir", type=Path,
                        default="output",
                        help="Path to a saved vocoder")

    parser.add_argument("--low_mem", action="store_true", help= \
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help= \
        "If True, audio won't be played.")
    args = parser.parse_args()
    print_args(args, parser)

    # 语音文件输出目录
    _out_dir = Path(args.wav_output_dir)
    _out_dir.mkdir(parents=True, exist_ok=True)

    print("检测环境配置...\n")

    print("正在准备编码器、合成器和解码器...")
    encoder.load_model(args.enc_model_fpath, device='cpu')
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    # vocoder.load_model(args.voc_model_fpath)

    ## Run a test
    print("正在用小样本检测配置...")
    # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
    # sampling rate, which may differ.
    # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
    # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
    # The sampling rate is the number of values (samples) recorded per second, it is set to
    # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
    # to an audio of 1 second.
    print("\t检测 编码器...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))

    # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
    # returns, but here we're going to make one ourselves just for the sake of showing that it's
    # possible.
    embed = np.random.rand(speaker_embedding_size)
    # Embeddings are L2-normalized (this isn't important here, but if you want to make your own
    # embeddings it will be).
    embed /= np.linalg.norm(embed)
    # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
    # illustrate that

    embeds = [embed, np.zeros(speaker_embedding_size)]  #speaker_embedding_size = 256
    texts = ["文本一", "文本二"]
    print("\t正在检测合成器... (loading the model will output a lot of text)")
    #为每一段文本生成一个梅尔频谱, 计算耗时
    mels = synthesizer.synthesize_spectrograms(texts, embeds)

    #把多个梅尔频谱合并
    mel = np.concatenate(mels, axis=1)

    #使用 audio 合并梅尔频谱
    generated_wav = audio.inv_mel_spectrogram(mel, hparams.hparams)
    print("通过检测! 可以开始合成语音了\n\n")

    ## Interactive speech generation
    num_generated = 0
    preprocessed_wav = encoder.preprocess_wav('../data/demo.mp3')
    print("成功加载demo语音")
    # Then we derive the embedding. There are many functions and parameters that the
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    embed = encoder.embed_utterance(preprocessed_wav)
    print("embedding创建完成")

    print("进入循环交互界面")
    while True:
        try:
            ## Generating the spectrogram
            text = input("请输入一小段文字:\n")
            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]

            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("梅尔频谱创建完成")
            print("正在合成波形图:")

            wav = synthesizer.griffin_lim(spec)
            ##  sd.play(wav, sample_rate)

            # generated_wav = audio.inv_mel_spectrogram(spec, hparams.hparams)
            fpath = _out_dir.joinpath("output_{}.wav".format(time.strftime('%Y%m%d_%H%M%S')))
            audio.save_wav(wav, fpath, synthesizer.sample_rate)


            # print(generated_wav.dtype)
            # librosa.output.write_wav(fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\n音频文件已保存到 %s\n\n" % fpath)
        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")