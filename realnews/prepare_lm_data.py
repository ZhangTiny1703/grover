"""
Turn a merged corpus into tfrecord files.

NOTE: You will want to do this using several processes. I did this on an AWS machine with 72 CPUs using GNU parallel
as that's where I had the deduplicated RealNews dataset.
"""
import argparse
import ujson as json
from sample.encoder import get_encoder, tokenize_for_grover_training, detokenize, sliding_window, create_int_feature
import random
import tensorflow as tf
import collections
import os
from tempfile import TemporaryDirectory

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'fold',0,
    'which fold we are on')
flags.DEFINE_integer(
    'num_folds',1,
    'Number of folds (corresponding to both the number of training files and the number of testing files)')
flags.DEFINE_integer(
    'seed',1337,
    'which seed to use')
flags.DEFINE_string(
    'base_fn','realnews_',
    'We will output files that are like {base_fn}_{n}.tfrecord for n in 0, ..., 1023')
flags.DEFINE_string(
    'input_fn',None,
    'Base filename to use. THIS MUST BE A LOCAL FILE.')
flags.DEFINE_integer(
    'max_seq_length',1024,
    'Max sequence length')
flags.DEFINE_bool(
    'add_extra_articles_to_end',True,
    'Whether to minimize padding by adding extra articles to the end')

random.seed(FLAGS.seed + FLAGS.fold)

encoder = get_encoder()


class S3TFRecordWriter(object):
    def __init__(self, fn):
        self.fn = fn
        if fn.startswith('s3://'):
            from boto3.s3.transfer import TransferConfig
            import boto3
            self.gclient = None
            self.s3client = boto3.client('s3',
                                         )
            self.storage_dir = TemporaryDirectory()
            self.writer = tf.python_io.TFRecordWriter(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.bucket_name, self.file_name = self.fn.split('s3://', 1)[1].split('/', 1)
        elif fn.startswith('gs://'):
            from google.cloud import storage
            self.s3client = None
            self.gclient = storage.Client()
            self.storage_dir = TemporaryDirectory()
            self.writer = tf.python_io.TFRecordWriter(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.bucket_name, self.file_name = self.fn.split('gs://', 1)[1].split('/', 1)

        else:
            self.s3client = None
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            self.writer = tf.python_io.TFRecordWriter(fn)

    def write(self, x):
        self.writer.write(x)

    def close(self):
        self.writer.close()

        if self.s3client is not None:
            from boto3.s3.transfer import TransferConfig
            config = TransferConfig(multipart_threshold=1024 * 25, max_concurrency=10,
                                    multipart_chunksize=1024 * 25, use_threads=True)
            self.s3client.upload_file(
                os.path.join(self.storage_dir.name, 'temp.tfrecord'),
                self.bucket_name,
                self.file_name,
                ExtraArgs={'ACL': 'public-read'}, Config=config,
            )
            self.storage_dir.cleanup()
        if self.gclient is not None:
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        print("CALLING CLOSE")
        self.close()


def article_iterator(encoder, final_desired_size=1025):
    """ Iterate through the provided filename + tokenize"""
    assert os.path.exists(FLAGS.input_fn)
    with open(FLAGS.input_fn, 'r') as f:
        for l_no, l in enumerate(f):
            if l_no % FLAGS.num_folds == FLAGS.fold:
                article = json.loads(l)
                article['input_ids'] = tokenize_for_grover_training(encoder, article, desired_size=final_desired_size,
                                                                    unconditional_prob=.35)
                article['inst_index'] = (l_no // FLAGS.num_folds)
                if article['inst_index'] < 100:
                    print('---\nINPUT{}. {}\n---\nTokens: {}\n'.format(article['inst_index'],
                                                                       detokenize(encoder, article['input_ids']),
                                                                       article['input_ids']
                                                                       ), flush=True)
                if len(article['input_ids']) == 0:
                    continue
                yield article


def _stream_from_buffer(buffer, current_desired_size, pad_token=0, add_articles_to_end=False):
    """ Combines short articles that are in a buffer """
    random.shuffle(buffer)
    i = 0
    while i < len(buffer):
        article = buffer[i]
        if add_articles_to_end:
            for article2add in buffer[(i + 1):]:
                i += 1
                article['input_ids'].append(encoder.padding)
                article['input_ids'].append(encoder.reset_context)
                article['input_ids'].extend(article2add['input_ids'])

                if len(article['input_ids']) >= current_desired_size:
                    article['input_ids'] = article['input_ids'][:current_desired_size]
                    break
        # print(f"YIELD FROM BUFFER {i}")

        # Pad to right length
        amount_to_pad = current_desired_size - len(article['input_ids'])
        article['input_ids'].extend([pad_token] * amount_to_pad)
        article['sub_index'] = 0
        yield article
        i += 1


def buffered_and_sliding_window_article_iterator(encoder, current_desired_size, final_desired_size=1025):
    """ We apply a sliding window to fix long sequences, and use a buffer that combines short sequences."""
    assert current_desired_size <= final_desired_size
    buffer = []
    for article in article_iterator(encoder, final_desired_size=final_desired_size):
        amount_to_pad = current_desired_size - len(article['input_ids'])

        if article['split'] == 'val' or amount_to_pad <= 0:
            for sub_index, sub_article in enumerate(sliding_window(article, max_seq_length=current_desired_size,
                                                                   pad_token=encoder.padding)):
                sub_article['sub_index'] = sub_index
                # print(f"AMT2PAD < 0 YIELD-{inst_index} sliding window {sub_index}", flush=True)
                yield sub_article
        else:
            # Buffer time.
            buffer.append(article)

        if len(buffer) % 100 == 0:
            yield from _stream_from_buffer(buffer,
                                           current_desired_size=current_desired_size,
                                           pad_token=encoder.padding,
                                           add_articles_to_end=FLAGS.add_extra_articles_to_end)
            buffer = []
    yield from _stream_from_buffer(buffer,
                                   current_desired_size=current_desired_size,
                                   pad_token=encoder.padding,
                                   add_articles_to_end=FLAGS.add_extra_articles_to_end)


# OK now write the tfrecord file
total_written = 0
train_file = FLAGS.base_fn + 'train{:04d}.tfrecord'.format(FLAGS.fold)
val_file = FLAGS.base_fn + 'val{:04d}.tfrecord'.format(FLAGS.fold)
with S3TFRecordWriter(train_file) as train_writer, S3TFRecordWriter(val_file) as val_writer:
    for article in buffered_and_sliding_window_article_iterator(encoder, current_desired_size=FLAGS.max_seq_length + 1,
                                                                final_desired_size=max(FLAGS.max_seq_length + 1, 1025)):
        writer2use = train_writer if article['split'] == 'train' else val_writer
        assert len(article['input_ids']) == (FLAGS.max_seq_length + 1)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(article['input_ids'])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writer2use.write(tf_example.SerializeToString())
        total_written += 1

        # DEBUG
        if article['inst_index'] < 5:
            print("~~~\nSubindex{}. Index {}. ARTICLE: {}\n---\nTokens: {}\n\n".format(article['sub_index'],
                                                                                       article['inst_index'],
                                                                                       detokenize(encoder,
                                                                                                  article['input_ids']),
                                                                                       article['input_ids']),
                  flush=True)
        if article['inst_index'] % 1000 == 0:
            print("{} articles, {} written".format(article['inst_index'], total_written), flush=True)
print("DONE UPLOADING", flush=True)
