# InceptoFormer

![alt text](https://github.com/SafwenNaimi/InceptoFormer/blob/main/architecture.png)


## Getting the code
You can download a copy of all the files in this repository by cloning the git repository.

`python -m pip install -U pip`

`pip install -r requirements.txt`


## Running Experiments

To run experiments, execute the following file.

`python main.py`

The algorithm will generate the following output files:

    ├── output(dir)
        ├── train_classifier_month_day(dir)   
            ├── hour_minutes(dir)
	            ├──  model.json: JSON file of the model.               
	            ├──  res_pat.csv: results of accuracy, sensitivity and specificity by patients.
                ├──  res_seg.csv: results of accuracy, sensitivity and specificity by segments.	                
                ├──  training_i.csv: training/validation loss and accuracy for the i_th folder (i = [1..10]).   
	            ├──  weights_i.hdf5 : weights of the model for the i_th folder (i = [1..10]).   
