# Emp_from_Landsat
Predicting Employment from Landsat data of villages



open tensorboard on local machine:
http://localhost:16006

![Inception-v3 Architecture](Results/graph.png)


Training:
To train vggm netork:
nohup python train.py > log,txt 2>&1 &

To visualize tensorboard run on vm:

ssh from local system using:
ssh -L 16006:127.0.0.1:6006 <username@ip>

run tensorboard on vm :
nohup tensorboard --logdir= <directory of saved model>       > tensorboard_out.txt 2>&1 &


Training network with learning rate of 1e-4 for 200 iterations and then 1e-5 gives following loss plot:

![Train_loss](Results/train_loss.png)

Evaluating model on test data:

To evaluate the model performance on test data:

nohup python test.py > eval_log.txt 2>&1 &

To plot confusion matrix:

python plot_confusion_matrix.py

Results:

Unnormalized Confusion matrix:

![Unnormalized_confusion_matrix](Results/unnormalized_confusion_matrix.png)


Normalized Confusion matrix:

![Normalized_confusion_matrix](Results/normalized_confusion_matrix.png)





