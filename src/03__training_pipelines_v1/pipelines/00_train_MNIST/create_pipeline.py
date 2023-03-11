from azureml.core import Workspace,Environment,ComputeTarget,Datastore,Dataset,Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import PipelineParameter,Pipeline
from azureml.pipeline.steps import PythonScriptStep
#############################################################################################
###### We create/get the Environment
#############################################################################################
ws=Workspace.from_config()
env_target = Environment.get(workspace=ws, name="torchenv")
compute_target = ComputeTarget(workspace=ws, name="cc-f296849-gpu")
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
        "--batch-size",         PipelineParameter(name="batch size", default_value=4),
        "--learning-rate",      PipelineParameter(name="Learning Rate", default_value=1e-4),
        "--scheduler-step",     PipelineParameter(name="Scheduler Steps", default_value=2),
        "--scheduler-gamma",     PipelineParameter(name="Scheduler Gamma", default_value=0.5),
        "--cluster" # We say to the script that we are in cluster mode...
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