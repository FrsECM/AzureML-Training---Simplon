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
RUN pip install '<URL HERE>'