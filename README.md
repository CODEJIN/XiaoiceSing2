# XiaoiceSing 2

This code is an implementation for XiaoiceSing 2. The algorithm is based on the following papers:
```
Wang, C., Zeng, C., & He, X. (2022). Xiaoicesing 2: A High-Fidelity Singing Voice Synthesizer Based on Generative Adversarial Network. arXiv preprint arXiv:2210.14666.
```

* Unfortunately, due to the limitations of the equipment, this repository was simply tested to see whether the code worked. So please note that this repository will be on the status of __Work In Progress__ for a long time.

# Structure
* Structure is based on the XiaoiceSing 2.
* I changed several hyper parameters and data type
    * One of mel or spectrogram is can be selected as a feature type.
    * Token type is changed from phoneme to grapheme.
    * Duration predictor and upsampler is based on the Gaussian upsampler.
    * By vocoder problem, I changed the sample rate from 48Khz of paper to 22.05Khz.

# Tested dataset
* [CSD Dataset](https://github.com/emotiontts/emotiontts_open_db/tree/master/Dataset/CSD)
    * A modified version of some data with differences between note and f0 through pitch tracking (unpublished).
* [AIHub Multi Singer Singing Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=465)

# Hyper parameters
Before proceeding, please set the pattern, inference, and checkpoint paths in [Hyper_Parameters.yaml](Hyper_Parameters.yaml) according to your environment.

* Feature_Type
    * Setting the feature type (`Mel` or `Spectrogram`).

* Sound
    * Setting basic sound parameters.

* Tokens
    * The number of Lyric token.

* Notes
    * The highest note value for embedding.

* Durations
    * The longest duration of note.
* Duration        
    * Equality set the strategy about syllable to grapheme.
        * When `True`, onset, nucleus, and coda have same length or ±1 difference.
        * When `False`, onset and coda have Consonant_Duration length, and nucleus has duration - 2 * Consonant_Duration.

* Encoder
    * Setting the encoder(embedding).

* Variance_Block
    * Setting the duration predictor

* Train
    * Setting the parameters of training.

* Inference_Batch_Size
    * Setting the batch size when inference

* Inference_Path
    * Setting the inference path

* Checkpoint_Path
    * Setting the checkpoint path

* Log_Path
    * Setting the tensorboard log path

* Use_Mixed_Precision
    * Setting using mixed precision

* Use_Multi_GPU
    * Setting using multi gpu
    * By the nvcc problem, Only linux supports this option.
    * If this is `True`, device parameter is also multiple like '0,1,2,3'.
    * And you have to change the training command also: please check  [multi_gpu.sh](./multi_gpu.sh).

* Device
    * Setting which GPU devices are used in multi-GPU enviornment.
    * Or, if using only CPU, please set '-1'. (But, I don't recommend while training.)

# Generate pattern

```
python Pattern_Generate.py [parameters]
```
## Parameters
* -csd
    * The path of children's song dataset
* -aihub
    * The path of AIHub Multi Singer Singing Dataset.
* -step
    * The note step that is explored when generating patterns.
    * The smaller step is, the more patterns are created in one song.
* -hp
    * The path of hyperparameter.
    
# Inference file path while training for verification.

* Inference_for_Training
    * There are three examples for inference.
    * It is midi file based script.

# Run

## Command

### Single GPU
```
python Train.py -hp <path> -s <int>
```

* `-hp <path>`
    * The hyper paramter file path
    * This is required.

* `-s <int>`
    * The resume step parameter.
    * Default is `0`.
    * If value is `0`, model try to search the latest checkpoint.

### Multi GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=32 python -m torch.distributed.launch --nproc_per_node=8 Train.py --hyper_parameters Hyper_Parameters.yaml --port 54322
```

* I recommend to check the [multi_gpu.sh](./multi_gpu.sh).