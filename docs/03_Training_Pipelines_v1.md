# Training pipelines

## Introduction
In this part, we'll learn how to build a training pipeline in AzureML.
In the first part we'll create our EntryPoint and debug it with vscode.
When it will be debugged, we'll learn how to create an environment where to execute this code.
Finally, we'll run this code as a Pipeline in AzureML.

## Create our Developpement Tree
First of all, we'll create our training pipeline in a script.
Create a new folder **pipeline** in your AzureML Workspace containing a "00_trainMNIST" directory.
In this folder we'll create :
- A Dockerfile
- A **scripts** directory
- A **train_mnist.py** file in this script directory.

Because we are in a public environment, there is no link with Artifactory.
In order to be able to use a custom library, we'll upload it's whl in a **library** directory.
- Create a **library** directory.
- Upload there the **libaicv.X.X.X.whl** file.

Your workspace should look like that :<br>
<img src="../assets/img/00026.png" alt="drawing" width="400"/>

We will create a new dev environment (vscode) :
```bash
conda create -n azml_training python=3.8
```
Reload the windows (Shift+Crtl+P) : <br>
<img src="../assets/img/00027.png" alt="drawing" width="400"/>

And select the environment :<br>
<img src="../assets/img/00028.png" alt="drawing" width="400"/>

Now, we are ready to start dev, but first, we'll do a short introduction about **Compute Instance** vs **Compute Clusters**.

## Computer Instance / Compute Cluster
A **compute instance** is a machine that can be use only by one user, it is schedulable in order to start and stop automatically and you can developp directly on it.<br>
A **compute cluster**, is a machine that can be use by all users, it is automatically scalable and you pay only the time of computation.<br>

Bellow a short summary of their capabilities :
|                      |Compute Instance | Compute Cluster |
|----------------------|-----------------|-----------------|
|MultiNode             |                 | &check;         |
|MultiUser             |                 | &check;         |
|AutoScale             |                 | &check;         |
|RealTime Debug        | &check;         |                 |
|Pipeline Compatibility| &check;         | &check;         |
|Launch & Forget       |                 | &check;         |

What you can see from this table, is that it is very convenient to :
- Use **Compute Instance** for debug and dev
- Use **Compute Cluster** for Execution

We'll start by creating our training script on our compute instance.

# MNIST Training - **Local**
First, we'll install our new library in the newly created pipeline.
As we are in AzureML, we want to use 
<div style="background-color:#0000EE22">
IN AZUREML TERMINAL<br>
</div>

```bash
# We install Pytorch CPU Version first...
(azureml_training) >> pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# Then we install libaicv
(azureml_training) >> pip install pipelines/00_train_MNIST/library/libaicv-0.1.0-py3-none-any.whl[azureml]
```

## Load data
<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> pipelines/00_train_MNIST/scripts/train_mnist.py
</div>

```python
from azureml.core import Workspace, Datastore,Dataset
import os

if __name__=='__main__':
    workspace = Workspace.from_config()
    datastore = Datastore.get(workspace, 'mnist_data')
    # We put in a list because we can mount more than one datastore at the same time. [(datastore, '/**'),(datastore_labels,'/**')]
    dataset_paths=[(datastore, '/**')]  
    dataset = Dataset.File.from_files(path=dataset_paths)
    mount_context = dataset.mount()
    mount_context.start()
    #######################################
    # We set our I/O Pathes
    #######################################
    input_path = mount_context.mount_point
    output_path = "results"
    classes = os.listdir(input_path)
    mount_context.stop()
    print('Terminé !')
```

In order to check that everything is working well, we can put a debugpoint on the line **mount_context.stop()**<br>
<img src="../assets/img/00029.png" alt="drawing" width="800"/>

Normally, you are able to see your 10 directories containing your images... If you want, you can test to train with a lot more images by comming back to previous steps (data).

## Run Training - **Local**
Now we'll use libaicv to perform a training..
We'll modify the previous code like that :
<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> pipelines/00_train_MNIST/scripts/train_mnist.py
</div>

```python
...
# Add New dependancies
from libaicv.classification import ClassificationDataset,ResNetClassifier,ClassificationTrainer
from libaicv.core import ACCELERATOR
import os

if __name__=="__main__":
    ...
    #######################################
    # We Create training...
    #######################################    
    dataset = ClassificationDataset.fromFolder(rootFolder=input_path,size=(28,28))
    dataset.split()
    model = ResNetClassifier(nChannels=3,clsCount=dataset.clsCount)
    # We set the trainer and all it's dependancies...
    trainer = ClassificationTrainer(model,root_logdir=args.output_dir,num_worker=os.cpu_count(),log_batch=False)
    trainer.set_StepScheduler(2,0.5)
    trainer.fit(dataset,batch_size=4,epochs=5,accelerator=ACCELERATOR.CPU)
    #######################################
    mount_context.stop()
    print('Terminé')
```
If it runs correctly, you should be able to see it : <br>
<img src="../assets/img/00030.png" alt="drawing" width="800"/>

## Tensorboard - **Local**
In previous pipeline, we've created a tensorboard logdir directly in our environment.
In order to watch the results, we'll have to create a tensorboard launch script....
- Create a new **launch_tensorboard_local.py**

<img src="../assets/img/00031.png" alt="drawing" width="300"/>

In this script write something like that :

<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> tensorboard/launch_tensorboard_local.py
</div>

```python
from azureml.tensorboard import Tensorboard

local_outputs = 'results'

tb = Tensorboard([],local_root=local_outputs,port=8009)
tb.start()
input('Type Enter to stop tensorboard...')
tb.stop()
```
Run this script, you should be able to see a link in the terminal :<br>
<img src="../assets/img/00032.png" alt="drawing" width="800"/>

If you **(Ctrl + click)** on the link, it will open a new tensorboard tab :<br>
<img src="../assets/img/00033.png" alt="drawing" width="800"/>

We can see that our gradient is high, we can try to reduce the learning rate to get better training results.

## Argparse - Parametrisation
Our script is running well ! Congratulations !

Now it's working well, we'll give it some parameters for :
- **Input Directory**
- **Output Directory**
- *Batch Size*
- *Epochs*
- *LearningRate*
- *SchedulingFreq*
- *SchedulingGamma*
- *cluster*
In order to do this, we'll use **argparse**, a python standard Library. At the beggining of our script, we'll add our parameters...

<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> pipelines/00_train_MNIST/scripts/train_mnist.py
</div>

```python
... # After imports...
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',default=None)
parser.add_argument('--output_dir',default="results")
parser.add_argument('--epochs',type=int,default=5)
parser.add_argument('--batch-size',type=int,default=4)
parser.add_argument('--learning-rate',type=float,default=1e-4)
parser.add_argument('--scheduler-step',type=int,default=2)
parser.add_argument('--scheduler-gamma',type=float,default=0.5)
# We add our debug mode...
parser.add_argument('--debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=True)

if __name__=='__main__':
    # We parse our Arguments
    args = parser.parse_args()
    ...
```
We'll create a **cluster** parameter. This parameter will be necessary because the way we mount our datastore is different in cluster or in compute instances...<br>
We'll modify our script to override **input_dir**.

The script will normally look like that :<br>
<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> pipelines/00_train_MNIST/scripts/train_mnist.py
</div>

```python
import os

from libaicv.classification import ClassificationDataset,ResNetClassifier,ClassificationTrainer
from libaicv.core import ACCELERATOR
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir',default=None)
parser.add_argument('--output-dir',default="results")
parser.add_argument('--epochs',type=int,default=5)
parser.add_argument('--batch-size',type=int,default=4)
parser.add_argument('--learning-rate',type=float,default=1e-4)
parser.add_argument('--scheduler-step',type=int,default=2)
parser.add_argument('--scheduler-gamma',type=float,default=0.5)
# We add our cluster mode...
parser.add_argument('--cluster', action='store_true')
parser.add_argument('--no-cluster', dest='cluster', action='store_false')
parser.set_defaults(cluster=False)


if __name__=='__main__':
    # We parse our Arguments
    args = parser.parse_args()
    if not args.cluster:
        from azureml.core import Workspace, Datastore,Dataset
        workspace = Workspace.from_config()
        datastore = Datastore.get(workspace, 'mnist_data')
        # We put in a list because we can mount more than one datastore at the same time. [(datastore, '/**'),(datastore_labels,'/**')]
        dataset_paths=[(datastore, '/**')]  
        dataset = Dataset.File.from_files(path=dataset_paths)
        mount_context = dataset.mount()
        mount_context.start()
        #######################################
        # We set our I/O Pathes
        #######################################
        args.input_dir = mount_context.mount_point
    #######################################
    # We Create training...
    #######################################    
    dataset = ClassificationDataset.fromFolder(rootFolder=args.input_dir,size=(28,28))
    dataset.split()
    model = ResNetClassifier(nChannels=3,clsCount=dataset.clsCount)
    # We set the trainer and all it's dependancies...
    trainer = ClassificationTrainer(model,root_logdir=args.output_dir,num_worker=os.cpu_count(),log_batch=False)
    trainer.set_StepScheduler(args.scheduler_step,args.scheduler_gamma)
    trainer.fit(dataset,batch_size=args.batch_size,epochs=args.epochs,accelerator=ACCELERATOR.CPU)
    #######################################
    if not args.cluster:
        mount_context.stop()
    
    print('Terminé !')
```
---
# MNIST Training - **Pipeline**
As we explained before, when you want to launch multiple experiments of the same training job, it's better to use pipelines in order to perform **Launch and Forget**.<br>
In this part, we'll create :
- A GPU Compute Cluster...
- An Execution environment (with custom library)
- A Launch Pipeline Script

## Create the GPU Cluster
In order to run our job on a cluster, we'll need to create a new gpu cluster.
For that, we'll go in **Compute** tab and add it manually.<br>
<img src="../assets/img/00035.png" alt="drawing" width="800"/>
We'll select this kind of machine if available :<br>
<img src="../assets/img/00036.png" alt="drawing" width="800"/>
If not, we'll use a one that is available with a GPU...<br>
<img src="../assets/img/00037.png" alt="drawing" width="800"/>
For the purpose of training, we'll set the **Idle before scale down** to 20000. It will let us 6 hours before scaling down the node after a job fails what is good when we developp.<br>
We'll name our machine "**cc-f\<Michelin ID\>-gpu"** .

## Create the Environment
The environement is the image of what will be the environment when we'll run our pipeline. There we'll declare our libraries and dependancies.<br>

The easiest way to create an environment is to use a **DockerFile**.<br>
For example, our base Dockerfile with GPU support will look like that :<br>

<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> pipelines/00_train_MNIST/Dockerfile
</div>

```docker
FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu116-py38-torch1121:biweekly.202211.2

RUN pip install 'ipykernel~=6.0' \
                'azureml-core==1.48.0' \
				'azureml-dataset-runtime==1.48.0' \
                'azureml-defaults==1.48.0' \
				'azure-ml==0.0.1' \
				'azure-ml-component==0.9.15.post2' \
                'azureml-mlflow==1.48.0' \
		        'azureml-contrib-services==1.48.0' \
		        'azureml-contrib-services==1.48.0' \
                'torch-tb-profiler~=0.4.0' \
				'py-spy==0.3.12' \
		        'debugpy~=1.6.3'

```

In order to declare the environment, we'll add a new script in our pipeline. We'll name it : **pipelines/00_train_MNIST/create_environment.py**<br>

<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> pipelines/00_train_MNIST/scripts/create_environment.py
</div>

```python
from azureml.core import Workspace,Environment
from azureml.core.conda_dependencies import CondaDependencies

BUILD = True

ws=Workspace.from_config()
# We get the existing base environment
new_env = Environment.from_dockerfile(name="torchenv",dockerfile="pipelines/00_train_MNIST/Dockerfile")
# We add our libaicv library and get the link.
library_url = Environment.add_private_pip_wheel(workspace=ws,file_path = "pipelines/00_train_MNIST/library/libaicv-0.1.0-py3-none-any.whl",exist_ok=True)
print('Library URL to add to your Dockerfile before build')
print(f">> {library_url}")
# Register the environment
if BUILD:
    new_env.register(workspace=ws)
    new_env.build(workspace=ws)
```

We'll first launch the script in order to know where will be stored our library. This way we'll be able to add it to our dockerfile.<br>
For that we'll need the url + the SaS token because the library is stored on a private storage... We'll generate it manually...
<img src="../assets/img/00038.png" alt="drawing" width="800"/><br>

And finally we'll add it to our Dockerfile.<br>

<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> pipelines/00_train_MNIST/scripts/Dockerfile
</div>

```dockerfile
....
# Install pip dependencies (libaicv)
RUN pip install 'https://stamlfdme2968492248aml.blob.core.windows.net/azureml/Environment/azureml-private-packages/libaicv-0.1.0-py3-none-any.whl?sp=r&st=2022-12-15T19:07:26Z&se=2032-12-16T03:07:26Z&spr=https&sv=2021-06-08&sr=c&sig=dE2qYcUGMd33D%2BceQkY1Emjf6QYhRff%2FCddYOTSFTMs%3D'
...
###

```
Run again your script with :
```python
BUILD=True
```
If you go in **Environment** tab, you'll see your new environment currently building :<br>
<img src="../assets/img/00034.png" alt="drawing" width="800"/>

When the image will be ready to use, the status will become **succeeded**.


## Create the Launch Script
Now we have :
- Our Source Script
- Our Environment
- Our Compute Target

It's time to create the launch script that will launch our training job !
In order to do this, we'll create a new **create_pipeline.py** script.<br>

<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> pipelines/00_train_MNIST/scripts/create_pipeline.py
</div>

```python
from azureml.core import Workspace,Environment,ComputeTarget
from azureml.core.runconfig import RunConfiguration

from azureml.core import Datastore,Dataset
from azureml.data import OutputFileDatasetConfig

from azureml.pipeline.core import PipelineParameter,Pipeline
from azureml.pipeline.steps import PythonScriptStep

from azureml.core import Experiment

#############################################################################################
###### We create/get the Environment
#############################################################################################
ws=Workspace.from_config()
env_target = Environment.get(workspace=ws, name="torchenv")
compute_target = ComputeTarget(workspace=ws, name="cc-f<Michelin ID>-gpu")
#--------------------------------------------------------------------------------------------
# We create the Run Context
gpu_run_config=RunConfiguration()
gpu_run_config.target=compute_target
gpu_run_config.environment=env_target

#############################################################################################
###### We Create the pipeline
#############################################################################################
# We get the datastores ...
datastore_data=Datastore.get(ws,'mnist_data')
datastore_default=Datastore.get_default(ws)
# We define Inputs / Outputs
input_data = Dataset.File.from_files(path=(datastore_data, f'/**'))
train_output = OutputFileDatasetConfig(destination = (datastore_default, f'mnist_results'),name=f'mnist_results')

train_step = PythonScriptStep(
    name="train_MNIST",
    script_name="scripts/train_mnist.py",
    source_directory='pipelines/00_train_MNIST',
    arguments=[
        # Dataset
        "--input-dir",         input_data.as_download(),
        "--output-dir",         train_output,
        # Training
        "--epochs",             PipelineParameter(name="Epochs", default_value=10),
        "--batch-size",         PipelineParameter(name="Batch size", default_value=4),
        "--learning-rate",      PipelineParameter(name="Learning Rate", default_value=2e-3),
        "--scheduler-step",     PipelineParameter(name="Scheduler Steps", default_value=2),
        "--scheduler-gamma",     PipelineParameter(name="Scheduler Gamma", default_value=0.5),
        "--cluster"
        ],
    compute_target=compute_target,
    runconfig=gpu_run_config,
    allow_reuse=False
)

#############################################################################################
###### We Launch the pipeline
#############################################################################################
train_pipeline=Pipeline(workspace=ws, steps=[train_step])
# Submit the pipeline to be run
pipeline_run1 = Experiment(ws, f'MNIST_Training').submit(train_pipeline)
pipeline_run1.display_name = f"MNIST_Training"
```

Before running this script, don't forget to change the accelerator in the launch script...<br>

<div style="background-color:#0000EE22">
IN AZUREML WORKSPACE<br>
>> pipelines/00_train_MNIST/scripts/train_mnist.py
</div>

```python
# If you want to add logs in azureml + tensorboard :
import os
os.environ['LIBAICV_AZUREML_LOGGING']="1"
...
trainer.fit(dataset,batch_size=args.batch_size,epochs=args.epochs,accelerator=ACCELERATOR.CUDA)
    ...
```

## Run Training - **Cluster**
If you launch the "**create_pipeline.py**" file, you should be able to see a new run in jobs.

<img src="../assets/img/00039.png" alt="drawing" width="800"/>

As you can see, the environment variable worked and allowed us to log our metrics in azureml directly....

But where is our tensorboard ?


## Tensorboard - **Cluster**
We'll create a new script in order to launch tensorboard a little differently.
The main difference is that we'll have to mount the storage before launching tensorboard.
- Copy/Paste our local tensorboard script and rename it "cluster".

Edit it like that => We'll mount our storage prior analyse.
```python
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
    tb = Tensorboard([],local_root=mount_context.mount_point,port=8009)
    tb.start()
    input('Type Enter to stop tensorboard...')
    tb.stop()
```

And now we can see our tensorboard !<br>
<img src="../assets/img/00041.png" alt="drawing" width="800"/>

---
[<< Back](../README.md)