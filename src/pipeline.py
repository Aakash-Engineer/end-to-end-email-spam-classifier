from data_ingestion import DataIngestion
from model import Model
from pathlib import Path


if __name__ == '__main__':
    # data ingestion
    data_ingestion = DataIngestion()
    data_ingestion.main()
    
    # model
    model = Model()
    model.main(train_data_path=Path('../artifacts/training/train.csv'), 
               save_model_path=Path('../models/model.pkl'))
