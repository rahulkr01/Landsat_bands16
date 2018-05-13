import numpy as np
from os import walk
import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
import vggm
from scipy import misc
from libtiff import TIFF
import sys
import numpy as np
from PIL import Image
import os
import pickle
import libtiff
libtiff.libtiff_ctypes.suppress_warnings()
import glob
import pandas as pd
import random


random.seed(2001)
path=os.getcwd()
resizeDim=200
nchannels=16
path = os.getcwd()
# image_path="/sampleImages1/"
image_path="/tifSingle/"

dirs=os.listdir(path+image_path)
print(dirs)
files=[]
for direc in dirs:
	file1=glob.glob(path+image_path+direc+"/*.tif")
	files.extend(file1)
N= len(files)
index_arr=np.arange(N)
index_arr=np.asarray(index_arr,dtype=np.int32)
random.shuffle(index_arr)
train_len=int(0.8*N)
# train_len=1000
train_files=index_arr[:train_len]
test_files=index_arr[train_len:]
df=pd.read_csv(path+image_path+"/Vill.csv")
village_code=df["Town/Village"].values
emp_label=df["Village_HHD_Cluster_EMP"].values
actual_labels= [ int(c.split(' ')[0].split('.')[0]) for c in emp_label]
s1 = pd.Series(actual_labels,index=list(village_code))

def get_batch_data():
	global train_files,files,s1
	i=0
	X=np.array([]).reshape((0,resizeDim,resizeDim, nchannels))
	Y=[]	
	j=0
	i=0
	k=0
	random.shuffle(train_files)

	for ind in train_files:
		tif = TIFF.open(files[ind], mode='r')
		image = tif.read_image()
		dataAll = np.array(image)
		if(dataAll.shape[0]>resizeDim or dataAll.shape[1]>resizeDim):
			continue

		village_code=int((files[ind].split('@')[3]).split('.')[0])
		val=0
		try:
			try:
				val=int(s1.loc[village_code])
			except:
				continue
		except:
			continue
		data=np.delete(dataAll,[11,12],axis=2)

		band2=data[:,:,1]
		band3=data[:,:,2]
		band4=data[:,:,3]
		band5=data[:,:,4]
		band6=data[:,:,5]
		band7=data[:,:,6]
		sum45=band4+band5
		sum35=band3+band5
		sum56=band5+band6
		sum57=band5+band7
		####ndvi
		sum45[sum45==0.0]=1.0
		ndvi=(band5-band4)/sum45
		####ndwi
		sum35[sum35==0.0]=1.0
		ndwi=(band3-band5)/sum35
		####ndbi
		sum56[sum56==0.0]=1.0
		ndbi=(band6-band5)/sum56
		####ui
		sum57[sum57==0.0]=1.0
		ui=(band7-band5)/sum57
		####evi
		complexDenom=(band5+6*band4-7.5*band2+1.0)
		complexDenom[complexDenom==0.0] = 1.0
		band4Denom= band4.copy()
		band4Denom[band4Denom==0.0]=1.0
		eviHelper=2.5*(band5/band4Denom)
		evi=eviHelper/complexDenom

		combinedData=np.dstack((data,ndvi,ndwi,ndbi,ui,evi))

		left=(resizeDim-combinedData.shape[0])//2
		right=resizeDim-combinedData.shape[0]-left
		up=(resizeDim-combinedData.shape[1])//2
		down=resizeDim-combinedData.shape[1]-up

		data1=np.lib.pad(combinedData,[(left,right),(up,down),(0,0)],'constant')
		data1=np.reshape(data1,(1,200,200,16))
		if np.isnan(data1).any():
			continue
		else:
			X=np.vstack((X,data1))
			Y.append(val)

		i+=1
        
		if i%(64)==0:
			X=np.asarray(X,dtype=np.float32)
			Y-=1
                        Y=np.asarray(Y,dtype=np.int32)
			dataset = (X, Y)
			return dataset

j=0
ind=0
def get_eval_data():
	global j
	global ind
	global test_files,files,s1

	X=np.array([]).reshape((0,resizeDim,resizeDim, nchannels))
	Y=[]
	while ind< len(test_files):
		ind=(ind+1)%len(test_files)
		tif = TIFF.open(files[test_files[ind]], mode='r')
		image = tif.read_image()
		dataAll = np.array(image)
		if(dataAll.shape[0]>resizeDim or dataAll.shape[1]>resizeDim):
			continue

		village_code=int((files[test_files[ind]].split('@')[3]).split('.')[0])
		val=0
		try:
			try:
				val=int(s1.loc[village_code])
			except:
				continue
		except:
			continue
		data=np.delete(dataAll,[11,12],axis=2)

		band2=data[:,:,1]
		band3=data[:,:,2]
		band4=data[:,:,3]
		band5=data[:,:,4]
		band6=data[:,:,5]
		band7=data[:,:,6]
		sum45=band4+band5
		sum35=band3+band5
		sum56=band5+band6
		sum57=band5+band7
		####ndvi
		sum45[sum45==0.0]=1.0
		ndvi=(band5-band4)/sum45
		####ndwi
		sum35[sum35==0.0]=1.0
		ndwi=(band3-band5)/sum35
		####ndbi
		sum56[sum56==0.0]=1.0
		ndbi=(band6-band5)/sum56
		####ui
		sum57[sum57==0.0]=1.0
		ui=(band7-band5)/sum57
		####evi
		complexDenom=(band5+6*band4-7.5*band2+1.0)
		complexDenom[complexDenom==0.0] = 1.0
		band4Denom= band4.copy()
		band4Denom[band4Denom==0.0]=1.0
		eviHelper=2.5*(band5/band4Denom)
		evi=eviHelper/complexDenom

		combinedData=np.dstack((data,ndvi,ndwi,ndbi,ui,evi))

		left=(resizeDim-combinedData.shape[0])//2
		right=resizeDim-combinedData.shape[0]-left
		up=(resizeDim-combinedData.shape[1])//2
		down=resizeDim-combinedData.shape[1]-up

		data1=np.lib.pad(combinedData,[(left,right),(up,down),(0,0)],'constant')
		data1=np.reshape(data1,(1,200,200,16))
		if np.isnan(data1).any():
			continue
		else:
			X=np.vstack((X,data1))
			Y.append(val)

		j+=1
		if j%(64)==0:
			X=np.asarray(X,dtype=np.float32)
			Y=np.asarray(Y,dtype=np.int32)
                        Y-=1
			dataset = (X, Y)
			return dataset
	return
k=0
ind=0
def get_eval_train_data():
	global k
	global ind
	global train_files,files,s1

	X=np.array([]).reshape((0,resizeDim,resizeDim, nchannels))
	Y=[]
	while ind < len(train_files):
		ind=(ind+1)%len(train_files)
		tif = TIFF.open(files[train_files[ind]], mode='r')
		image = tif.read_image()
		dataAll = np.array(image)
		if(dataAll.shape[0]>resizeDim or dataAll.shape[1]>resizeDim):
			continue

		village_code=int((files[train_files[ind]].split('@')[3]).split('.')[0])
		val=0
		try:
			try:
				val=int(s1.loc[village_code])
			except:
				continue
		except:
			continue
		data=np.delete(dataAll,[11,12],axis=2)

		band2=data[:,:,1]
		band3=data[:,:,2]
		band4=data[:,:,3]
		band5=data[:,:,4]
		band6=data[:,:,5]
		band7=data[:,:,6]
		sum45=band4+band5
		sum35=band3+band5
		sum56=band5+band6
		sum57=band5+band7
		####ndvi
		sum45[sum45==0.0]=1.0
		ndvi=(band5-band4)/sum45
		####ndwi
		sum35[sum35==0.0]=1.0
		ndwi=(band3-band5)/sum35
		####ndbi
		sum56[sum56==0.0]=1.0
		ndbi=(band6-band5)/sum56
		####ui
		sum57[sum57==0.0]=1.0
		ui=(band7-band5)/sum57
		####evi
		complexDenom=(band5+6*band4-7.5*band2+1.0)
		complexDenom[complexDenom==0.0] = 1.0
		band4Denom= band4.copy()
		band4Denom[band4Denom==0.0]=1.0
		eviHelper=2.5*(band5/band4Denom)
		evi=eviHelper/complexDenom

		combinedData=np.dstack((data,ndvi,ndwi,ndbi,ui,evi))

		left=(resizeDim-combinedData.shape[0])//2
		right=resizeDim-combinedData.shape[0]-left
		up=(resizeDim-combinedData.shape[1])//2
		down=resizeDim-combinedData.shape[1]-up

		data1=np.lib.pad(combinedData,[(left,right),(up,down),(0,0)],'constant')
		data1=np.reshape(data1,(1,200,200,16))
		if np.isnan(data1).any():
			continue
		else:
			X=np.vstack((X,data1))
			Y.append(val)

		k+=1
		if k%(64)==0:
			X=np.asarray(X,dtype=np.float32)
			Y=np.asarray(Y,dtype=np.int32)
			Y-=1
                        dataset = (X, Y)
			print(k)
			return dataset

	return



def main(unused_argv):

	orig_stdout = sys.stdout
	orig_stderr=sys.stderr
	f = open('Results/eval_out.txt', 'w')
	f1= open('Results/eval_warning.txt','w')
	sys.stdout = f
	sys.stderr= f1




	# Create the Estimator
	num_epochs=1
	batch_size=64
	nsteps=100
	mnist_classifier = tf.estimator.Estimator( model_fn=vggm.cnn_model_fn, model_dir=path+"/Model")

			

	test_batch_size=batch_size
	total_test_batch= len(test_files)//test_batch_size
	avg=0
	predicted_y=np.array([])
	actual_y=np.array([])

	for i in range(total_test_batch):


	# Evaluate the model and print results
		evalX,evalY=get_eval_data()
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		  x=evalX,
		  y=evalY,
		  num_epochs=1,
		  shuffle=False)
		eval_results = mnist_classifier.evaluate(input_fn= eval_input_fn)
		y2= list(mnist_classifier.predict(input_fn=eval_input_fn))
		y1= np.array([ p["classes"] for p in y2 ])
		tupl= eval_results["accuracy"]
		avg+=tupl
		predicted_y=np.hstack((predicted_y,y1))
		actual_y=np.hstack((actual_y,evalY))
	

	print("test_accuracy: ",avg/total_test_batch)
	arr=np.vstack((actual_y,predicted_y)).T
	np.savetxt('Results/predicted_labels.txt',arr)


	total_train_batch=train_len//test_batch_size
	avg=0
	# for i in range(total_train_batch):


	# # Evaluate the model and print results
	# 	evalX,evalY=get_eval_train_data()
	# 	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	# 	  x=evalX,
	# 	  y=evalY,
	# 	  num_epochs=1,
	# 	  shuffle=False)
	# 	eval_results = mnist_classifier.evaluate(input_fn= eval_input_fn)
	# 	avg+=eval_results["accuracy"]

	# # print(eval_results)
	# print("train_accuracy: ",avg/total_train_batch)
	sys.stdout = orig_stdout
	sys.stderr= orig_stderr
	f.close()
	f1.close()


if __name__ == "__main__":
	tf.app.run()
