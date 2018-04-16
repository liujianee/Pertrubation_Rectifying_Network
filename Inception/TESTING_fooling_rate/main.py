import sys
import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config

config = None

def main(_):
  config, unparsed = get_config()
  prepare_dirs(config)

  #rng = np.random.RandomState(config.random_seed)
  #tf.set_random_seed(config.random_seed)

  trainer = Trainer(config)
  #writer = tf.summary.FileWriter('dbg_logs', sess.graph)
  #[n.name for n in sess.graph.as_graph_def().node]
  #save_config(config.model_dir, config)


  if config.is_train:
    trainer.train()
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()

if __name__ == "__main__":
  tf.app.run(main=main)




