import matplotlib
matplotlib.use('Agg') # because librosa.display includes matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import librosa

import librosa.display
from scipy.optimize import nnls
import os
import time
from utils import mkdir, read_via_scipy, get_config, magnitude2waveform, spectrum2magnitude

### basic settings
# estimated time: 7sec * num_styles * num_pieces
sr = 22050
num_styles = 5#10
hei, wid = 256, 256
num_pieces = 5#None # output first k pieces of spectra, each last 3 seconds
D_phase = None # without phase information, the util will do phase estimation

### directory & file names
gen_dir = './generated_features'
ver_id = 'song1'

spectra_dir = gen_dir + '/' + ver_id + '/'
print('spectra_dir = ', spectra_dir)

source_wav = './raw_audios/raw_audio_piano/' + 'QingTianJayChou.wav'
print(source_wav)
is_overwrite = True # reconstructing costs a lots of time, only run it if necessary

config_name = gen_dir + '/' + ver_id + '_config.yaml'
print("config_name = {}".format(config_name))

#config = get_config(config_name)
### settings
config = {
    # basic parameters
    'sr': 22050,
    'fft_size': 2048,
    'hop_length': 256,
    'input_type': 'exp',  # power, dB with ref_dB, p_log, exp with exp_b. it's input of training data
    'is_mel': True,

    # for spectra
    'n_mels': 256,
    'exp_b': 0.3,
    'ref_dB': 1e-5,

    # for cepstrum
    'dct_type': 2,
    'norm': 'ortho',

    # for slicing and overlapping
    'audio_samples_frame_size': 77175,  # 3.5sec * sr
    'audio_samples_hop_length': 77175,
    'output_hei': 256,
    'output_wid': 302,  # num_output_frames = 1 + (77175/hop_length256)

    # to decide number of channels
    'use_phase': False,  # only True without mel
    'is_multi': False,  # if true, there would be three resolutions
    'use_ceps': True,
    'use_d_spec': True,
    'd_spec_type': 'attack',  # mode: all, decay, or attack
    'use_spec_enve': True,

    'num_digit': 4
}

outdir = 'generated_audios' + '/' + ver_id + '/'
print("out_dir = {}".format(outdir))
mkdir(outdir)

source_wav = None ###
if source_wav is not None:
    print('phase information is from {}'.format(source_wav))
    y, sr = read_via_scipy(source_wav)
    y = y / np.max(np.abs(y))
    wav_name = outdir+'phase_info_source'+'.wav'
    librosa.output.write_wav(wav_name, y, sr)
    # extract the phase information
    D_mag, D_phase = librosa.magphase(librosa.stft(y, n_fft=config['fft_size'], hop_length=config['hop_length']))

for style in range(1):
    #suffix_file_name = '*'+'_style_'+str(style).zfill(2)+'.npy'
    #glob_files = spectra_dir #+ suffix_file_name
    # gen_dir + ver_id + 'piece0000_style_00.npy'
    #files = sorted(glob.glob(glob_files))
    #num_files = len(files)
    #print('{}: contains {} files'.format(glob_files, len(files)))
    #if num_files==0:
    #    continue
    #if num_pieces == None: # then process all spectra
    #    num_pieces = num_files

    num_pieces = len(os.listdir(spectra_dir))
    
    ret = np.zeros((hei, wid*num_pieces), dtype='float32') # concatenate the spectra
    cnt = 0
    for file in os.listdir(spectra_dir):
        if cnt>=num_pieces:
            break
        
        x = np.load(os.path.join(spectra_dir, file)) # x.dtype = 'float32'
        if len(x.shape)==3:
            # in latest codes,  x.shape should be [num_ch, 256. 256]
            ret[:,cnt*256:(cnt+1)*256] = x[0] # only use spectrogram
        else:
            # x.shape = [256, 256]
            ret[:,cnt*256:(cnt+1)*256] = x 
        cnt += 1
    print('shape of npy file: {}'.format(x.shape))
    if not np.isfinite(ret).all():
        print('Error !!!\nThe spectrogram is nan')
    
    wav_name = outdir+'style_'+str(style).zfill(2)+'.wav'
    print(wav_name)
    
    png_name = wav_name[:-4]+'.png'
    plt.figure(1, figsize=(7*3, 7/302*256*1))
    plt.clf()
    librosa.display.specshow(ret[:,:256*3], y_axis='mel', x_axis='time', hop_length=config['hop_length'])
    plt.savefig(png_name, dpi='figure', bbox_inches='tight')
    
    if (is_overwrite==False) and os.path.isfile(wav_name):
        print('{} already exists'.format(wav_name))
        continue
    
    print('*'*5+'reconstructing magnitude'+'*'*5)
    
    st = time.time()
    mag = spectrum2magnitude(ret, config)
    ed = time.time()
    print('nnls average cost {} seconds for {} pieces'.format((ed-st)/num_pieces, num_pieces))
    print(mag.shape, mag.dtype)
    print('*'*5+'reconstructing waveform'+'*'*5)
    audio = magnitude2waveform(mag, config, D_phase)
    print(audio.shape, audio.dtype)
    if not np.isfinite(audio).all():
        print('Error !!!\nThe audio is nan')

    if np.max(np.abs(audio)) > 0.0:
        # normalize the output audio
        norm_audio = audio/np.max(np.abs(audio))
    else:
        norm_audio = audio
        wav_name = wav_name[:-4]+'_notNorm.wav'
    librosa.output.write_wav(wav_name, norm_audio, sr)
