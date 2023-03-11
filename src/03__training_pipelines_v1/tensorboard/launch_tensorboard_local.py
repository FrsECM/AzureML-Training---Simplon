from azureml.tensorboard import Tensorboard

local_outputs = 'results'

tb = Tensorboard([],local_root=local_outputs,port=8009)
tb.start()
input('Type Enter to stop tensorboard...')
tb.stop()