FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu116-py38-torch1121:biweekly.202211.2

# Install MS Dependancies (pypi)
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

# Install pip dependencies (libaicv) => Artifactory Library
RUN pip install 'https://stamlfdme2968492248aml.blob.core.windows.net/azureml/Environment/azureml-private-packages/libaicv-0.0.2+4.g95e9738-py3-none-any.whl?sp=r&st=2022-12-15T19:07:26Z&se=2032-12-16T03:07:26Z&spr=https&sv=2021-06-08&sr=c&sig=dE2qYcUGMd33D%2BceQkY1Emjf6QYhRff%2FCddYOTSFTMs%3D'