# Not tested, correct in case of error.
# Install repository
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
# Install dependancies
sudo apt-get install libfuse3-dev fuse3
# Install Blobfuse2
sudo apt install blobfuse2