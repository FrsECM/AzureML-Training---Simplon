sudo mkdir /media/aml_data
sudo mkdir /tmp/.azcache
# Replace myuser by your user, in my case f296849
sudo chown azureuser /media/aml_data
sudo chown azureuser /tmp/.azcache
sudo blobfuse2 mount all /media/aml_data/ --config-file="/home/azureuser/cloudfiles/code/Users/francois.ponchon/blobfuse2/config/mnist_data.yaml"