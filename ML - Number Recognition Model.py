# =================================================  Settings  =================================================
#  weights_method
# =================================================
#  load starting weights                        = 1
#  load existing weights                        = 2
#  random weights                               = 3
#  he initialization                            = 4
#  he normal initialization                     = 5
# =================================================
#
#
# one_dataset
# =================================================
# a single dataset with all data                = 1
# file name: dataset_single_file
# ( var: cutoff will split the dataset for training 
# and testing )
# two files, one training, one testing          = 0
# file name: dataset_train_file, dataset_test_file
# =================================================
#
#
# save_data
# =================================================
# save weights when training and at the end     = 1
# don't                                         = 0
# =================================================
# 
#
# create_kaggle_result_csv
# =================================================
# run a prediction on Kaggle's testing file     = 1
# don't                                         = 0
# file name: kaggle_test_file
# =================================================
#
#
# display_error_matrix_of_trained_data
# =================================================
# run a prediction on your training dataset
# and see what errors were made. 
# example. 1 was predicted as 9                 = 1
# don't                                         = 0
# =================================================
# 

# File Names & Path
path = "C:\\Users\\User\\"
dataset_single_file = "Train PLUS Augmented data.csv"
dataset_train_file = "dataset_train.csv"
dataset_test_file = "dataset_test.csv"
kaggle_test_file = "test.csv"

weights_method = 2
one_dataset = 0
save_data = 1
create_kaggle_result_csv = 0    
display_error_matrix_of_trained_data = 0

# Parameters
alpha = 0.1
epochs = 50000
cutoff = 5000
layer_nodes = 235
output_layer_nodes = 10


# =================================================     End of Settings     =================================================


import pandas as pd
import cupy as cp
import time 
from datetime import datetime, timedelta

class Display:
   def time_stamp(starttime):
      if starttime:
         print("Code started: ", str(starttime.hour).zfill(2) + ':'+ str(starttime.minute).zfill(2) + ':' + str(starttime.second).zfill(2))
         starttime = datetime.now()
         print("Code ended: ", str(starttime.hour).zfill(2) + ':'+ str(starttime.minute).zfill(2) + ':' + str(starttime.second).zfill(2))
      else:
         starttime = datetime.now()
         print("Code started: ", str(starttime.hour).zfill(2) + ':'+ str(starttime.minute).zfill(2) + ':' + str(starttime.second).zfill(2))
         return starttime

   def final_accuracy(W1, b1, W2, b2):
      _, _, testedActivationW2 = Training.forward_propagation(W1, b1, W2, b2, final_PixelsTestData)
      final_accuracy = cp.mean( num_Labels_test == cp.argmax(testedActivationW2, 0))
      print("Accuracy on testing set:", final_accuracy)

   def results_board(epoch, timestart, alpha, accuracy):
      timestamp = int(time.time() - timestart)
      expectedfinish = int(timestamp/epoch*epochs) if epoch else 0
      print("")
      print("Tested on:", total_rows - training_rows) 
      print("Trained on:", training_rows)
      print("Total dataset:", total_rows)
      print("Alpha:", alpha)
      print("Nodes:", layer_nodes)
      print("Time running:", str(timedelta(seconds=timestamp)).zfill(8), "/", str(timedelta(seconds=expectedfinish)).zfill(8))
      print("Epochs:", epoch, "/", epochs)
      print("Accuracy on training set:", accuracy)
   
   def error_matrix(num_Labels_test, testedActivationW2):
      result_of_test = cp.argmax(testedActivationW2, 0)
      dic = {}
      for i in range(10):
         newDic = {}
         for j in range(10):
            newDic[j] = 0
         dic[i] = newDic

      for i in range(total_rows):
         if num_Labels_test[i] != result_of_test[i]:
            a = int(num_Labels_test[i])
            b = int(result_of_test[i])
            dic[a][b] += 1
      
      for i in range(10):
         print(i, dic[i])
      



class DataPreparation:
   def save_WeightsAndBiases(save_list):
      for name, weightOrBiasesData in save_list:
         cp.savetxt(path + name, weightOrBiasesData, delimiter=',')

   def one_hot(pxLabels):
      one_hot_matrix = cp.zeros((pxLabels.size, int(pxLabels.max()+1)))
      one_hot_matrix[cp.arange(pxLabels.size), pxLabels] = 1
      one_hot_matrix = one_hot_matrix.T
      return one_hot_matrix

   def convert_to_usable_data(data_to_process):
      num_Labels_to_int = data_to_process[0].astype(int)
      oneHot_Labels_to_process = DataPreparation.one_hot(num_Labels_to_int)  
      final_PixelsData = data_to_process[1:] / maxNumberInAllArrays
      return num_Labels_to_int, oneHot_Labels_to_process, final_PixelsData

   def create_WeightsAndBiases_Random(layer_nodes, output_layer_nodes, cols):
      W1 = cp.random.rand(layer_nodes, cols-1) - 0.5
      b1 = cp.random.rand(layer_nodes, 1) - 0.5
      W2 = cp.random.rand(output_layer_nodes, layer_nodes) - 0.5
      b2 = cp.random.rand(output_layer_nodes, 1) - 0.5
      if save_data:
         DataPreparation.save_WeightsAndBiases([["starting_W1.csv",W1], ["starting_b1.csv",b1], ["starting_W2.csv",W2], ["starting_b2.csv", b2]])
      return W1, b1, W2, b2

   def create_WeightsAndBiases_He_Initialization(layer_nodes, output_layer_nodes, cols):
      W1 = cp.random.randn(layer_nodes, cols-1) * cp.sqrt(2/(layer_nodes*(cols-1)))
      b1 = cp.random.randn(layer_nodes, 1) * cp.sqrt(2/(layer_nodes))
      W2 = cp.random.randn(output_layer_nodes, layer_nodes) * cp.sqrt( 2/ (layer_nodes*layer_nodes))
      b2 = cp.random.randn(output_layer_nodes, 1) * cp.sqrt(2/(layer_nodes))
      if save_data:
         DataPreparation.save_WeightsAndBiases([["starting_W1.csv",W1], ["starting_b1.csv",b1], ["starting_W2.csv",W2], ["starting_b2.csv", b2]])
      return W1, b1, W2, b2

   def create_WeightsAndBiases_He_Normal_Initialization(layer_nodes, output_layer_nodes, cols):
      normal_limit = cp.sqrt(2 / float(cols-1))
      W1 = cp.random.normal(0.0, normal_limit, size=(layer_nodes, cols-1))
      b1 = cp.zeros((layer_nodes, 1))
      W2 = cp.random.normal(0.0, normal_limit, size=(output_layer_nodes, layer_nodes))
      b2 = cp.zeros((output_layer_nodes, 1))
      if save_data:
         DataPreparation.save_WeightsAndBiases([["starting_W1.csv",W1], ["starting_b1.csv",b1], ["starting_W2.csv",W2], ["starting_b2.csv", b2]])
      return W1, b1, W2, b2



class ActivationFunctions:
   def ReLU(Z):
      return cp.maximum(Z, 0)

   def ReLU_deriv(Z):
      return Z > 0

   def softplus(Z):
      return cp.log(1 + cp.exp(Z))

   def softplus_deriv(Z):
      A = cp.exp(Z)
      return A / (1 + A)

   def leaky_ReLU(Z):
      return cp.maximum(Z, Z * 0.01)

   def leaky_ReLU_deriv(Z):
      return cp.where(Z > 0, 1, 0.01)
   
   def softmax(Z): 
      A = cp.exp(Z)
      return A / sum(A)
   


class Training:
   def forward_propagation(W1, b1, W2, b2, pxData):
      afterW1 = W1.dot(pxData) + b1                  
      afterActivationW1 = ActivationFunctions.ReLU(afterW1)              
      afterW2 = W2.dot(afterActivationW1) + b2      
      afterActivationW2 = ActivationFunctions.softmax(afterW2)         
      return afterW1, afterActivationW1, afterActivationW2

   def backward_propagation(afterW1, afterActivationW1, afterActivationW2, W2, pxData, oneHot_Labels_train): 
      pVSl = (oneHot_Labels_train - afterActivationW2) * -2   
      backprobW2 = pVSl.dot(afterActivationW1.T) / rows          
      backprobB2 = cp.sum(pVSl, axis=1).reshape(pVSl.shape[0],1) / rows                            
      
      pVSl_ReLuDeriv = W2.T.dot(pVSl) * ActivationFunctions.ReLU_deriv(afterW1)
      backprobW1 = pVSl_ReLuDeriv.dot(pxData.T) / rows        
      backprobB1 = cp.sum(pVSl_ReLuDeriv, axis=1).reshape(pVSl_ReLuDeriv.shape[0],1) / rows
      return backprobW1, backprobB1, backprobW2, backprobB2

   def update_WeightsAndBiases(W1, b1, W2, b2, backprobW1, backprobB1, backprobW2, backprobB2, alpha):
      W1 = W1 - (alpha * backprobW1) 
      b1 = b1 - (alpha * backprobB1) 
      W2 = W2 - (alpha * backprobW2) 
      b2 = b2 - (alpha * backprobB2) 
      return W1, b1, W2, b2

   def start(W1, b1, W2, b2, pxData, pxLabels, oneHot_Labels_train, alpha, iteration):
      timestart = time.time()
      for epoch in range(iteration):
         afterW1, afterActivationW1, afterActivationW2 = Training.forward_propagation(W1, b1, W2, b2, pxData)
         backprobW1, backprobB1, backprobW2, backprobB2 = Training.backward_propagation(afterW1, afterActivationW1, afterActivationW2, W2, pxData, oneHot_Labels_train)
         W1, b1, W2, b2 = Training.update_WeightsAndBiases(W1, b1, W2, b2, backprobW1, backprobB1, backprobW2, backprobB2, alpha)
         accuracy = cp.mean( pxLabels == cp.argmax(afterActivationW2, 0))

         if display_error_matrix_of_trained_data:
            Display.error_matrix(pxLabels, afterActivationW2)
            break

         if accuracy > 0.9999:
            break

         if epoch and epoch % 2 == 0:
            Display.results_board(epoch, timestart, alpha, accuracy)

            if cutoff > 0 or one_dataset == 0:
               Display.final_accuracy(W1, b1, W2, b2)
            
            # Save progress
            if epoch % 50 == 0 and save_data:
               DataPreparation.save_WeightsAndBiases([["W1.csv", W1], ["b1.csv", b1], ["W2.csv", W2], ["b2.csv", b2]])
               print("---> Saved")
            
      Display.time_stamp(keepStarttime)
      return W1, b1, W2, b2

class Kaggle:
   def result(W1, b1, W2, b2, testing_rows):
      result = [["ImageId","Label"]]
      for index in range(testing_rows):
         _, _, afterActivationW2 = Training.forward_propagation(W1, b1, W2, b2, kaggle_test_data[:, index, None])
         predicted_label = cp.argmax(afterActivationW2, 0)
         result.append([index+1, predicted_label[0]])
      pd.DataFrame(result).to_csv(path + "Kaggle result - Number Recognition.csv",  index=False, header=False)
      print("Finished")




#####  Execution  #####
if __name__ ==  '__main__':   

   ## Load Data
   rawData = cp.array(pd.read_csv(path + dataset_single_file))
   rows, cols = rawData.shape
   maxNumberInAllArrays = cp.amax(rawData)

   if one_dataset:
      trainingData = rawData[cutoff:rows].T
      testingData = rawData[:cutoff].T 
   else:
      cutoff = 0
      input_datatrain = cp.array(pd.read_csv(path + dataset_train_file))
      input_datatest = cp.array(pd.read_csv(path + dataset_test_file))
      trainingData = input_datatrain.T
      testingData = input_datatest.T


   ## Weights
   if weights_method == 1:
      W1 = cp.loadtxt(path + "starting_W1.csv", delimiter=',')
      b1 = cp.loadtxt(path + "starting_b1.csv", delimiter=',').reshape(-1,1)
      W2 = cp.loadtxt(path + "starting_W2.csv", delimiter=',')
      b2 = cp.loadtxt(path + "starting_b2.csv", delimiter=',').reshape(-1,1)
   elif weights_method == 2:
      W1 = cp.loadtxt(path + "W1.csv", delimiter=',')
      b1 = cp.loadtxt(path + "b1.csv", delimiter=',').reshape(-1,1)
      W2 = cp.loadtxt(path + "W2.csv", delimiter=',')
      b2 = cp.loadtxt(path + "b2.csv", delimiter=',').reshape(-1,1)
   elif weights_method == 3:
      W1, b1, W2, b2 = DataPreparation.create_WeightsAndBiases_Random(layer_nodes, output_layer_nodes, cols)
   elif weights_method == 4:
      W1, b1, W2, b2 = DataPreparation.create_WeightsAndBiases_He_Initialization(layer_nodes, output_layer_nodes, cols)
   elif weights_method == 5:
      W1, b1, W2, b2 = DataPreparation.create_WeightsAndBiases_He_Normal_Initialization(layer_nodes, output_layer_nodes, cols)



   if cutoff > 0 or one_dataset == 0:
      num_Labels_test, oneHot_Labels_test, final_PixelsTestData = DataPreparation.convert_to_usable_data(testingData)
      _, testing_rows = oneHot_Labels_test.shape
   else:
      testing_rows = 0

   if create_kaggle_result_csv:
      kaggle_test_data = cp.array(pd.read_csv(path + kaggle_test_file))
      kaggle_test_data = kaggle_test_data.T
      kaggle_cols, kaggle_rows  = kaggle_test_data.shape
      kaggle_test_data = kaggle_test_data / maxNumberInAllArrays
      Kaggle.result(W1, b1, W2, b2, kaggle_rows)

   else:
      num_Labels_train, oneHot_Labels_train, final_PixelsTrainData = DataPreparation.convert_to_usable_data(trainingData)
      keepStarttime = Display.time_stamp(0)
      training_rows = num_Labels_train.shape[0]
      total_rows = training_rows + testing_rows
      W1, b1, W2, b2 = Training.start(W1, b1, W2, b2, final_PixelsTrainData, num_Labels_train, oneHot_Labels_train, alpha, epochs+1)

      if save_data:
         DataPreparation.save_WeightsAndBiases([["W1.csv", W1], ["b1.csv", b1], ["W2.csv", W2], ["b2.csv", b2]])