from libaicv.classification import ClassificationDataset,ResNetClassifier,ClassificationTrainer
from libaicv.core import ACCELERATOR
import os,argparse
os.environ['LIBAICV_AZUREML_LOGGING']="1"
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
    trainer.fit(dataset,batch_size=args.batch_size,epochs=args.epochs,accelerator=ACCELERATOR.CUDA)
    #######################################
    if not args.cluster:
        mount_context.stop()    
    print('Terminé !')
