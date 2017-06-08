"""Tool to generate benchmark files for the audio alignment feature of Audacity.

This tool will process each pair of audio files in each subdirectory
of the input directory, and create test cases. Namely, it will apply
the following effects, to simulate difference recording devices:
- lowpass filter
- high-pass filters
- volume change
- (slight) speed change.
and it will also cut the second file in multiple pieces.

This tool invokes ffmpeg on the command line in order to apply all the
transformations.

Usage:
python3 src/effects/generate_alignment_benchmark.py  <benchmark_input_dir> <benchmark_output_dir>

"""

import os
import os.path
import shutil
import sys
import subprocess
import collections
import random
import pathlib

# Audio files less than this duration will be ignored.
MIN_AUDIO_LENGTH_SEC = 60
# Don't generate the clock drift test for files less than this
# duration.
MIN_AUDIO_LENGTH_SEC_FOR_DRIFT_TEST = 60 * 7
# For very long audio files of at least this duration, only generate a
# clock drift test. Don't generate other test cases.
MIN_AUDIO_LENGTH_SEC_ONLY_FULL_DRIFT_TEST = 30 * 60

FFPROBE_COMMAND = 'ffprobe'
FFMPEG_COMMAND = 'ffmpeg'

EXPECTED_FILENAME = 'expected'

LOW_PASS_OPTIONS = ['-af', 'lowpass']
HIGH_PASS_OPTIONS = ['-af', 'highpass=f=1000']
REDUCE_VOLUME_OPTIONS = ['-af', 'volume=-20dB']

# If this is true, and an output audio file already exists, don't
# regenerate it.
SKIP_IF_OUTPUT_EXISTS = True

def SpeedupOptions(filename, factor):
    rate = GetSampleRate(filename)
    new_rate = int(rate * factor)
    speedup = [
        '-af', 'asetrate=%d,aresample=resampler=soxr' % (new_rate,),
        '-ar',
        str(rate)]
    return speedup


def GetSampleRate(filename):
    cmd = [
        FFPROBE_COMMAND,
        '-v',
        'error',
        '-select_streams',
        'a:0',
        '-show_entries',
        'stream=sample_rate',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        filename]
    print('running', cmd)
    completed = subprocess.check_output(cmd)
    try:
        rate = int(completed)
    except ValueError:
        raise ValueError('Could not get sample rate of %s, got: %s' % (filename, completed))
    return rate


def GetDuration(filename):
    # TODO: replace with .run(stdout=subprocess.PIPE, check=TRUE)
    completed = subprocess.check_output([FFPROBE_COMMAND,
                    '-v',
                    'error',
                    '-show_entries',
                    'format=duration',
                    '-of',
                        'default=noprint_wrappers=1:nokey=1',
                        filename])
    try:
        duration = float(completed) #with .run(): float(completed.stdout)
    except ValueError:
        # completed.stdout
        raise ValueError('Could not get duration of %s, got: %s' % (filename, completed))
    return duration

def CheckMinDuration(filename):
    d = GetDuration(filename)
    if d < MIN_AUDIO_LENGTH_SEC:
        raise ValueError('File %s duration is %d, less than minimum required '
                         'duration %d' % (filename, d, MIN_AUDIO_LENGTH_SEC))
    return d

def FfmpegCommand(filename, outdir, ffmpeg_options):
    out_filename = os.path.join(
        outdir,
        os.path.splitext(os.path.basename(filename))[0] + '.flac')
    cmd = [FFMPEG_COMMAND,
            '-i',
           filename,
            '-acodec', 'flac'] + ffmpeg_options + [
                '-y', '-nostats', '-loglevel', '0',
                out_filename]
    return (cmd, out_filename)

Transform = collections.namedtuple('Transform',
                                   ['name', 'command',
                                    'expected_shift',
                                    'drift_multiplier_applied'])

def RunTransform(transform, filename_a, filename_b, output_dir):
    print('Processing transform', transform.name,
          'for', filename_a, filename_b)
    outdir = os.path.join(output_dir, transform.name)
    os.makedirs(outdir, exist_ok=True)
    full_command, out_filename_b = FfmpegCommand(filename_b, outdir, transform.command)
    if not SKIP_IF_OUTPUT_EXISTS or not os.path.isfile(out_filename_b):
        print('running', full_command)
        subprocess.check_call(full_command)

    copy_command, out_filename_a = FfmpegCommand(filename_a, outdir, [])
    if not SKIP_IF_OUTPUT_EXISTS or not os.path.isfile(out_filename_a):
        print('running', copy_command)
        subprocess.check_call(copy_command)
    return (outdir, out_filename_a, out_filename_b)


def RunSplits(filename_a, filename_b, output_dir, drift_multiplier=1,
              chunk_size_sec=60, max_num_chunks=10):
    CHUNK_SIZE_SEC = chunk_size_sec
    MAX_NUM_CHUNKS = max_num_chunks
    b_duration = GetDuration(filename_b)
    transforms = []
    for begin in range(0, int(b_duration) - CHUNK_SIZE_SEC, CHUNK_SIZE_SEC):
        end = begin + CHUNK_SIZE_SEC
        # In the presence of drift, assume the track will be aligned
        # according to its center.

        # if the b track is short, assume that its speed won't be
        # changed, and that it is aligned according to its center. If
        # it's long, its speed will be adjusted.
        shift_adjust = ((begin + end) * (drift_multiplier - 1) / 2
                        if CHUNK_SIZE_SEC < 60*5
                        else begin * (drift_multiplier - 1))
        transforms.append(Transform(
            '%d_%d' % (begin, end),
            ['-ss', str(begin), '-t', str(CHUNK_SIZE_SEC)],
            expected_shift=begin + shift_adjust,
            drift_multiplier_applied=drift_multiplier))
    random.seed(7)
    if len(transforms) > MAX_NUM_CHUNKS:
        # Keep first and last, this is where we can have potential issues.
        transforms = ([transforms[0]] +
                      random.sample(transforms[1:-1], MAX_NUM_CHUNKS - 2) +
                      [transforms[-1]])
    transforms.append(Transform('full', [], expected_shift=0,
                                drift_multiplier_applied=drift_multiplier))

    for transform in transforms:
        outdir, _, _ = RunTransform(transform, filename_a, filename_b, output_dir)
        with open(os.path.join(outdir, EXPECTED_FILENAME), 'w') as expected:
            expected.write('%f\n%f\n' % (transform.expected_shift, 1/drift_multiplier))
        # TODO: also exchange filenames, this way expected shift will be negative.


def RunEffects(filename_a, filename_b, output_dir):
    CheckMinDuration(filename_a)
    d_sec = CheckMinDuration(filename_b)

    transforms = []
    if d_sec < MIN_AUDIO_LENGTH_SEC_ONLY_FULL_DRIFT_TEST:
        transforms.extend([Transform('no_transform', [], None, 1),
                           Transform('low_pass', LOW_PASS_OPTIONS, None, 1),
                           Transform('high_pass', HIGH_PASS_OPTIONS, None, 1),
                           Transform('volume', REDUCE_VOLUME_OPTIONS, None, 1)])
    if d_sec >= MIN_AUDIO_LENGTH_SEC_FOR_DRIFT_TEST:
        drift = 1.001 if d_sec < MIN_AUDIO_LENGTH_SEC_ONLY_FULL_DRIFT_TEST else 1.0001
        transforms.append(
            Transform('clock_drift_%.4f' % drift, SpeedupOptions(filename_b, drift), None, drift))

    for transform in transforms:
        outdir, out_filename_a, out_filename_b = RunTransform(
            transform, filename_a, filename_b, output_dir)
        RunSplits(out_filename_a, out_filename_b, outdir,
                  transform.drift_multiplier_applied,
                  60 if d_sec < MIN_AUDIO_LENGTH_SEC_ONLY_FULL_DRIFT_TEST else 15 * 60)

def GenerateBenchmark(input_dir, output_dir):
    path = pathlib.Path(input_dir)
    for d in path.iterdir():
        if not d.is_dir():
            continue
        print('Processing', path)
        audio_files = sorted(x for x in d.iterdir()
                             if x.is_file() and x.suffix in ('.wav', '.flac', '.mp3'))
        if len(audio_files) != 2:
            raise ValueError('Expected exactly 2 files in %s, got %d' % (d, len(audio_files)))
        RunEffects(str(audio_files[0]), str(audio_files[1]),
                   os.path.join(output_dir, os.path.basename(str(d))))

if __name__ == '__main__':
    # input dir, output dir.
    GenerateBenchmark(sys.argv[1], sys.argv[2])
