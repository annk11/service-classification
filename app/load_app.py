import pickle

with open('./model/multinomial_nb_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)