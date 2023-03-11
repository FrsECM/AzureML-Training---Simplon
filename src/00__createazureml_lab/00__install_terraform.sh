pip install azure-cli
wget https://releases.hashicorp.com/terraform/1.3.5/terraform_1.3.5_linux_amd64.zip -O terraform.zip
unzip terraform.zip
sudo mv terraform /usr/local/bin
rm terraform.zip

# If your CLI is correctly defined :
terraform init