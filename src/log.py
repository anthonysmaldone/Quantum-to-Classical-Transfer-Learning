import os
import datetime
import csv

def prepare_log(hyperparameters):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("./output_cnn", exist_ok=True)
    file_name = os.path.join("./output_cnn", f"data_{current_datetime}_{hyperparameters['notes']}.csv")

    model_save_dir = f'./saved_models_cnn/{current_datetime}'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    with open(file_name, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows([list(hyperparameters.keys())])
        csv_writer.writerows([list(hyperparameters.values())])
        csv_writer.writerows([["Epoch", "Model Type", "Train Loss", "Train AUC", "Test Loss", "Test AUC"]])

    return file_name, current_datetime, model_save_dir

def training_log(file_name, epoch_data):
    with open(file_name, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows([epoch_data])
