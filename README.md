# Emp_from_Landsat
Predicting Employment from Landsat data of villages

To train vggm netork:
nohup python train.py > log,txt 2>&1 &

To evaluate the model performance on test data:
nohup python test.py > eval_log.txt 2>&1 &

To plot confusion matrix:
python plot_confusion_matrix.py

To visualize tensorboard run on vm:

ssh from local system using:
ssh -L 16006:127.0.0.1:6006 <username@ip>

run tensorboard on vm :
nohup tensorboard --logdir= <directory of saved model>       > tensorboard_out.txt 2>&1 &

open tensorboard on local machine:
http://localhost:16006