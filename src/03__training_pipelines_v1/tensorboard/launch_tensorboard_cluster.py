from azureml.core import Workspace,Datastore,Dataset
from azureml.tensorboard import Tensorboard
import os

cluster_outputs = 'mnist_results'
workspace = Workspace.from_config()
datastore = Datastore.get_default(workspace)

dataset_paths=[(datastore, '/**')] 
dataset = Dataset.File.from_files(path=dataset_paths)

with dataset.mount() as mount_context:
    result_paths = os.path.join(mount_context.mount_point,cluster_outputs)
    tb = Tensorboard([],local_root=result_paths,port=8009)
    tb.start()
    input('Type Enter to stop tensorboard...')
    tb.stop()